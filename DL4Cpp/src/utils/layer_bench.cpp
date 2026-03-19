

#include "utils/layer_bench.hpp"
#include "check.hpp"
#include "utils/log.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

namespace ctl {

namespace bench {

LayerTimeLogging::LayerTimeLogging(std::string_view layer_name,
                                   std::string_view layer_type)
    : layer_name_(layer_name), layer_type_(layer_type),
      start_time_(Time::now()) {
  auto &collector = LayerTimeStatsSingleton::SingletonInstance();
  auto &map_mutex = LayerTimeStatsSingleton::GetMapMutex();

  std::lock_guard<std::mutex> lock(map_mutex);
  const std::string name_str(layer_name);
  if (collector.find(name_str) == collector.end()) {
    collector.insert({name_str, std::make_unique<LayerTimeStats>(
                                    0, name_str, std::string(layer_type_))});
  }
}

LayerTimeLogging::~LayerTimeLogging() {
  const auto end_time = Time::now();
  const auto duration_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                            start_time_)
          .count();

  auto &collector = LayerTimeStatsSingleton::SingletonInstance();
  auto &map_mutex = LayerTimeStatsSingleton::GetMapMutex();
  LayerTimeStats *target_states = nullptr;

  {
    std::lock_guard<std::mutex> lock(map_mutex);
    auto iter = collector.find(std::string(layer_name_));
    if (iter != collector.end()) {
      target_states = iter->second.get();
    }
  }
  if (target_states != nullptr) {
    target_states->duration_time_.fetch_add(duration_time,
                                            std::memory_order_relaxed);
  } else {
    LOG(ERROR) << "LayerTimeLogging: Cannot find layer " << layer_name_
               << " in collector";
  }
}

void LayerTimeLogging::SummaryLogging() {
  auto &collector = LayerTimeStatsSingleton::SingletonInstance();
  auto &map_mutex = LayerTimeStatsSingleton::GetMapMutex();

  std::lock_guard<std::mutex> lock(map_mutex);
  long total_time_costs = 0;

  for (const auto &[layer_name, layer_time_stats] : collector) {
    CHECK(layer_time_stats != nullptr);
    const long time_cost =
        layer_time_stats->duration_time_.load(std::memory_order_relaxed);

    total_time_costs += time_cost;

    if (time_cost != 0) {
      LOG(INFO) << "Layer name: " << layer_name << "\t"
                << "layer type: " << layer_time_stats->layer_type_ << "\t"
                << "time cost: " << time_cost << "ms";
    }
  }
  LOG(INFO) << "Total time costs: " << total_time_costs << "ms";
}


} // namespace bench
} // namespace ctl