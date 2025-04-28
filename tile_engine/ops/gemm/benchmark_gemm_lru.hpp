// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <vector>
#include <list>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <thread>
#include <algorithm>
#include <cassert>

enum class Metric
{
    LATENCY,
    TFLOPS,
    BANDWIDTH
};

struct GemmProblem
{
    int split_k;
    int m, n, k;
    int stride_a, stride_b, stride_c;

    bool operator==(const GemmProblem& other) const
    {
        return split_k == other.split_k && m == other.m && n == other.n && k == other.k &&
               stride_a == other.stride_a && stride_b == other.stride_b &&
               stride_c == other.stride_c;
    }

    std::string to_string() const
    {
        return "split_k=" + std::to_string(split_k) + ", m=" + std::to_string(m) +
               ", n=" + std::to_string(n) + ", k=" + std::to_string(k) + ", strides=[" +
               std::to_string(stride_a) + "," + std::to_string(stride_b) + "," +
               std::to_string(stride_c) + "]";
    }
};

namespace std {
template <>
struct hash<GemmProblem>
{
    int operator()(const GemmProblem& p) const
    {
        return hash<int>()(p.split_k) ^ (hash<int>()(p.m) << 1) ^ (hash<int>()(p.n) << 2) ^
               (hash<int>()(p.k) << 3) ^ (hash<int>()(p.stride_a) << 4) ^
               (hash<int>()(p.stride_b) << 5) ^ (hash<int>()(p.stride_c) << 6);
    }
};
} // namespace std

struct KernelInstance
{
    std::string name;
    GemmProblem problem;
    float latency;
    float tflops;
    float bandwidth;

    bool operator==(const KernelInstance& other) const { return problem == other.problem; }

    std::string to_string() const
    {
        std::ostringstream oss;
        oss << name << " | " << problem.to_string() << " | Latency: " << std::fixed
            << std::setprecision(2) << latency << " ms"
            << " | TFLOPS: " << tflops << " | Bandwidth: " << bandwidth << " GB/s";
        return oss.str();
    }
};

// ---------- InstanceManager ----------
class InstanceManager
{
    private:
    std::vector<KernelInstance> instances_;
    std::atomic<size_t> current_version_{0};
    mutable std::mutex mutex_;

    public:
    void update_instances(const std::vector<KernelInstance>& instances)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        instances_ = instances;
        current_version_.fetch_add(1, std::memory_order_release);
    }

    std::vector<KernelInstance> get_instances_snapshot() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return instances_;
    }

    size_t current_version() const { return current_version_.load(std::memory_order_acquire); }
};

struct CacheKey
{
    size_t version;
    Metric metric;
    GemmProblem problem;

    bool operator==(const CacheKey& other) const
    {
        return version == other.version && metric == other.metric && problem == other.problem;
    }
};

namespace std {
template <>
struct hash<CacheKey>
{
    size_t operator()(const CacheKey& key) const
    {
        return hash<size_t>()(key.version) ^ (hash<int>()(static_cast<int>(key.metric)) << 1) ^
               (hash<GemmProblem>()(key.problem) << 2);
    }
};
} // namespace std

class BestInstanceCache
{
    private:
    struct CacheNode
    {
        CacheKey key;
        KernelInstance value;
    };

    using ListType = std::list<CacheNode>;
    using MapType  = std::unordered_map<CacheKey, typename ListType::iterator>;

    ListType lru_list_;
    MapType cache_map_;
    size_t max_size_  = 1000;
    size_t threshold_ = 1000;
    mutable std::mutex mutex_;

    void maintain_cache()
    {
        while(cache_map_.size() > max_size_)
        {
            auto& last_node = lru_list_.back();
            cache_map_.erase(last_node.key);
            lru_list_.pop_back();
        }
    }

    template <typename MetricFunc>
    KernelInstance find_best_impl(const std::vector<KernelInstance>& instances,
                                  MetricFunc&& metric_func)
    {
        return *std::min_element(
            instances.begin(), instances.end(), [&](const auto& a, const auto& b) {
                return metric_func(a) < metric_func(b);
            });
    }

    public:
    void set_threshold(size_t threshold) { threshold_ = threshold; }
    void set_max_size(size_t size) { max_size_ = size; }

    size_t get_cache_size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_map_.size();
    }

    const KernelInstance&
    get_best(const InstanceManager& manager, Metric metric, const GemmProblem& problem)
    {
        const size_t current_ver = manager.current_version();
        const CacheKey key{current_ver, metric, problem};

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if(auto it = cache_map_.find(key); it != cache_map_.end())
            {
                lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
                return it->second->value;
            }
        }

        const auto instances = manager.get_instances_snapshot();

        if(manager.current_version() != current_ver)
        {
            return get_best(manager, metric, problem);
        }

        std::vector<KernelInstance> filtered;
        std::copy_if(instances.begin(),
                     instances.end(),
                     std::back_inserter(filtered),
                     [&problem](const auto& inst) { return inst.problem == problem; });

        if(filtered.empty())
        {
            throw std::runtime_error("No instances for problem: " + problem.to_string());
        }

        KernelInstance best;
        const size_t num_elements = filtered.size();

        if(num_elements >= threshold_)
        {
            const unsigned hw_threads  = std::thread::hardware_concurrency();
            const unsigned num_threads = std::min<unsigned>(hw_threads ? hw_threads : 2,
                                                            static_cast<unsigned>(num_elements));

            std::vector<std::thread> threads(num_threads);
            std::vector<KernelInstance> local_bests(num_threads);
            const size_t chunk_size = (num_elements + num_threads - 1) / num_threads;

            auto metric_selector = [metric](const KernelInstance& inst) {
                switch(metric)
                {
                case Metric::LATENCY: return inst.latency;
                case Metric::TFLOPS: return -inst.tflops;
                case Metric::BANDWIDTH: return -inst.bandwidth;
                default: return 0.0f;
                }
            };

            for(unsigned i = 0; i < num_threads; ++i)
            {
                threads[i] = std::thread([&, i] {
                    auto start = filtered.begin() + i * chunk_size;
                    auto end   = (i == num_threads - 1) ? filtered.end() : start + chunk_size;
                    local_bests[i] =
                        *std::min_element(start, end, [&](const auto& a, const auto& b) {
                            return metric_selector(a) < metric_selector(b);
                        });
                });
            }

            for(auto& t : threads)
                t.join();
            best = *std::min_element(
                local_bests.begin(), local_bests.end(), [&](const auto& a, const auto& b) {
                    return metric_selector(a) < metric_selector(b);
                });
        }
        else
        {
            switch(metric)
            {
            case Metric::LATENCY:
                best = find_best_impl(filtered, [](const auto& inst) { return inst.latency; });
                break;
            case Metric::TFLOPS:
                best = find_best_impl(filtered, [](const auto& inst) { return -inst.tflops; });
                break;
            case Metric::BANDWIDTH:
                best = find_best_impl(filtered, [](const auto& inst) { return -inst.bandwidth; });
                break;
            }
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);

            if(manager.current_version() != current_ver)
            {
                return get_best(manager, metric, problem);
            }

            lru_list_.emplace_front(CacheNode{key, best});
            cache_map_[key] = lru_list_.begin();
            maintain_cache();
        }

        return lru_list_.front().value;
    }
};
