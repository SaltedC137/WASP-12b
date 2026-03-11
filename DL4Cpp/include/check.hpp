#ifndef LOGGING_HPP_
#define LOGGING_HPP_


#include <cstdlib>
#include <iostream>
#include <sstream>


#if defined(__GNUC__) || defined(__clang__)
    #define LIKELY(x) __builtin_expect(!!(x), 1)
    #define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define LIKELY(x) (x)
    #define UNLIKELY(x) (x)
#endif


namespace dlc_inf {

class FMessageLogger {
public:
  FMessageLogger(const char* file, int line) {
    stream_ << "[FATAL ERROR] at " << file << ":" << line << "\n";
  }

  ~FMessageLogger() {
    std::cerr << stream_.str() << std::endl;
    std::abort();
  }

  std::ostream &stream() { return stream_; }
  
private:
  std::ostringstream stream_;
};

class FMessageVoidify {
public:
  FMessageVoidify() = default;
  void operator&(std::ostream& stream){}
};
} // namespace dlc_inf



#define CHECK(conditions)                                                      \
  LIKELY(conditions)                                                           \
  ? (void)0                                                                    \
  : dlc_inf::FMessageVoidify() &                                               \
          dlc_inf::FMessageLogger(__FILE__, __LINE__).stream()                 \
              << "Check failed: " #conditions " "


#define CHECK_EQ(val1, val2)                                                   \
  LIKELY((val1) == (val2))                                                     \
  ? (void)0                                                                    \
  : dlc_inf::FMessageVoidify() &                                               \
          dlc_inf::FMessageLogger(__FILE__, __LINE__).stream()                 \
              << "Check failed: " #val1 " == " #val2 " (" << (val1) << " vs "  \
              << (val2) << ") "


#define CHECK_LT(val1, val2)                                                   \
  LIKELY((val1) < (val2))                                                      \
  ? (void)0                                                                    \
  : dlc_inf::FMessageVoidify() &                                               \
          dlc_inf::FMessageLogger(__FILE__, __LINE__).stream()                 \
              << "Check failed: " #val1 " < " #val2 " (" << (val1) << " vs "   \
              << (val2) << ") "


#define CHECK_LE(val1, val2)                                                   \
  LIKELY((val1) <= (val2))                                                     \
  ? (void)0                                                                    \
  : dlc_inf::FMessageVoidify() &                                               \
          dlc_inf::FMessageLogger(__FILE__, __LINE__).stream()                 \
              << "Check failed: " #val1 " <= " #val2 " (" << (val1) << " vs "  \
              << (val2) << ") "


#define CHECK_GE(val1, val2)                                                   \
  LIKELY((val1) >= (val2))                                                     \
  ? (void)0                                                                    \
  : dlc_inf::FMessageVoidify() &                                               \
          dlc_inf::FMessageLogger(__FILE__, __LINE__).stream()                 \
              << "Check failed: " #val1 " >= " #val2 " (" << (val1) << " vs "  \
              << (val2) << ") "

#endif