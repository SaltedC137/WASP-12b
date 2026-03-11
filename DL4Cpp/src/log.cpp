/**
 * @file log.cpp
 * @author Aska Lyn
 * @date 2026-03-11 19:35:36
 */

#include "log.hpp"
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <mutex>

namespace dlc_inf {


LogLevel SystemLogLevel = LogLevel::INFO;

LogMessage::LogMessage(LogLevel level, const char* file, int line) : level_(level) {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm* tm_now = std::localtime(&time_t_now);

    const char* level_str = "UNKNOWN";
    switch (level) {
        case LogLevel::DEBUG:   level_str = "DEBUG"; break;
        case LogLevel::INFO:    level_str = "INFO"; break;
        case LogLevel::WARNING: level_str = "WARN"; break;
        case LogLevel::ERROR:   level_str = "ERROR"; break;
        case LogLevel::FATAL:   level_str = "FATAL"; break;
    }

    stream_ << "[" << level_str << "] "
            << std::put_time(tm_now, "%Y-%m-%d %H:%M:%S") 
            << " [" << file << ":" << line << "] ";
}

LogMessage::~LogMessage() {
    static std::mutex log_mutex;
    {
        std::lock_guard<std::mutex> lock(log_mutex);
        if (level_ == LogLevel::FATAL || level_ == LogLevel::ERROR) {
            std::cerr << stream_.str() << std::endl;
        } else {
            std::cout << stream_.str() << std::endl;
        }
    }

    if (level_ == LogLevel::FATAL) {
        std::abort();
    }
}

std::ostream& LogMessage::stream() { return stream_; }

} // namespace dlc_inf