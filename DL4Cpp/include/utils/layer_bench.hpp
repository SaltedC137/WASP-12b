
/**
 * @file layer_bench.hpp
 * @author Aska Lyn
 * @brief A performance benchmarking utility for measuring layer execution time
 * @details This header provides RAII-based timing instrumentation for deep
 * learning layers. It uses a singleton pattern to collect timing statistics
 * across multiple layer executions, enabling performance profiling and
 * bottleneck identification in neural network forward/backward passes.
 * @date 2026-03-19
 */

#ifndef LAYER_BENCH_HPP
#define LAYER_BENCH_HPP

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace ctl {
namespace bench {

using Time = std::chrono::steady_clock; ///< Type alias for steady clock

/**
 * @brief Structure to hold timing statistics for a single layer
 * @details Stores the accumulated execution time and identification
 *          information for a specific layer instance. Uses atomic for
 *          thread-safe duration updates.
 */
struct LayerTimeStats {

  /**
   * @brief Construct a new Layer Time Stats object
   * @param duration_time Initial duration time in microseconds (or clock ticks)
   * @param layer_name The name identifier of the layer
   * @param layer_type The type of the layer (e.g., "Conv2D", "Dense")
   * @details Initializes the stats with the provided layer information.
   *          The duration_time_ is atomic for thread-safe accumulation.
   */
  explicit LayerTimeStats(long duration_time, std::string layer_name,
                          std::string layer_type)
      : duration_time_(duration_time), layer_name_(std::move(layer_name)),
        layer_type_(std::move(layer_type)) {}

  std::atomic<long> duration_time_{0}; ///< Accumulated execution time
  std::string layer_name_;             ///< Unique name identifier of the layer
  std::string layer_type_;             ///< Type classification of the layer
};

/// Type alias for the collector map that stores all layer statistics
using LayerTimeStatsCollector =
    std::unordered_map<std::string, std::unique_ptr<LayerTimeStats>>;

/**
 * @brief Singleton class to manage global layer timing statistics
 * @details Provides thread-safe access to a global map of layer statistics.
 *          Uses the Meyers' Singleton pattern with function-local static
 *          variables for lazy initialization and thread safety (C++11+).
 */
class LayerTimeStatsSingleton {
public:
  /**
   * @brief Get the singleton instance of the statistics collector
   * @return LayerTimeStatsCollector& Reference to the global collector map
   * @details Returns a reference to the static collector map. The map
   *          is created on first access and persists for the program lifetime.
   *          Thread-safe due to C++11 static initialization guarantees.
   */
  static LayerTimeStatsCollector &SingletonInstance() {
    static LayerTimeStatsCollector instance;
    return instance;
  }

  /**
   * @brief Get the mutex for protecting the collector map
   * @return std::mutex& Reference to the global map mutex
   * @details Returns a reference to the static mutex used for synchronizing
   *          access to the collector map. Callers must lock this mutex
   *          before modifying the map structure (inserting/erasing entries).
   */
  static std::mutex &GetMapMutex() {
    static std::mutex map_mutex;
    return map_mutex;
  }

  /**
   * @brief Clear all collected layer statistics
   * @details Removes all entries from the global collector map.
   *          Thread-safe access to the collector map.
   */
  static void ClearTimeStats() {
    auto &map_mutex = GetMapMutex();
    auto &collector = SingletonInstance();
    std::lock_guard<std::mutex> lock(map_mutex);
    collector.clear();
  }
};

/**
 * @brief RAII class for automatic layer execution timing
 * @details This class uses the RAII (Resource Acquisition Is Initialization)
 *          pattern to automatically measure layer execution time. Construction
 *          records the start time, and destruction calculates the elapsed time
 *          and accumulates it into the global statistics. Simply create an
 *          instance at the beginning of a layer's forward/backward pass.
 *
 *          Example usage:
 *          @code
 *          void Conv2D::forward() {
 *              LayerTimeLogging logging("conv1", "Conv2D");
 *              // ... layer computation ...
 *          } // Timing automatically recorded when logging goes out of scope
 *          @endcode
 */
class LayerTimeLogging {
public:
  /**
   * @brief Construct a new Layer Time Logging object
   * @param layer_name The name identifier of the layer being timed
   * @param layer_type The type of the layer being timed
   * @details Records the current time as the start point for timing.
   *          The layer will be registered in the global statistics
   *          collector if it doesn't already exist.
   */
  explicit LayerTimeLogging(std::string_view layer_name,
                            std::string_view layer_type);

  /**
   * @brief Destructor - records elapsed time and updates statistics
   * @details Calculates the time elapsed since construction and adds
   *          it to the layer's accumulated duration in the global
   *          statistics. Thread-safe access to the collector map.
   */
  ~LayerTimeLogging();

  /**
   * @brief Print a summary of all collected layer statistics
   * @details Iterates through all recorded layers and outputs their
   *          timing information to stdout. Includes total time, average
   *          time per call (if call count is tracked), and percentage
   *          breakdown. Useful for performance profiling reports.
   */
  static void SummaryLogging();

private:
  std::string layer_name_; ///< Name of the timed layer
  std::string layer_type_; ///< Type of the timed layer
  std::chrono::steady_clock::time_point start_time_; ///< Timing start point
};

} // namespace bench
} // namespace ctl
#endif