/**
 * @file check.hpp
 * @author Aska Lyn
 * @brief Logging and assertion utilities for DL4Cpp library
 * @details This header provides a CHECK macro system similar to Google's glog library.
 *          When a CHECK condition fails, the program logs a fatal error with file location
 *          and condition details, then aborts execution. Useful for catching programming
 *          errors and validating assumptions during development and debugging.
 * @date 2026-03-04 11:10:27
 */

#ifndef LOGGING_HPP_
#define LOGGING_HPP_


#include <iostream>
#include <sstream>


/* Compiler-specific branch prediction hints */
#if defined(__GNUC__) || defined(__clang__)
    #define LIKELY(x) __builtin_expect(!!(x), 1)
    #define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define LIKELY(x) (x)
    #define UNLIKELY(x) (x)
#endif


namespace dlc_inf {

/**
 * @brief Logger class for fatal error messages
 * @details Constructs an error message with file and line information,
 *          then aborts program execution when destroyed.
 */
class FMessageLogger {
public:
    /**
     * @brief Construct a fatal error logger
     * @param file The source file name where the error occurred
     * @param line The line number where the error occurred
     * @details Initializes the error message with location information.
     */
    FMessageLogger(const char* file, int line);

    /**
     * @brief Destructor - outputs the message and aborts
     * @details Writes the accumulated message to stderr and terminates
     *          the program using std::abort().
     */
    ~FMessageLogger();

    /**
     * @brief Get the underlying output stream
     * @return std::ostream& Reference to the internal string stream
     * @details Allows appending additional context to the error message.
     */
    std::ostream &stream();

private:
    std::ostringstream stream_;  ///< Internal stream for building the message
};

/**
 * @brief Voidify helper for ternary operator type matching
 * @details This class is used internally to make the CHECK macro work
 *          with the ternary operator by providing a compatible type
 *          for both branches.
 */
class FMessageVoidify {
public:
    FMessageVoidify() = default;
    
    /**
     * @brief Consume the ostream and discard the result
     * @param stream The ostream to consume
     * @details This operator is used to complete the CHECK macro expression
     *          without producing a value.
     */
    void operator&(std::ostream& stream);
};

} // namespace dlc_inf


/**
 * @brief Check a condition and abort if it fails
 * @param conditions The boolean expression to check
 * @details If the condition is false, logs a fatal error with the condition
 *          text and source location, then terminates the program.
 *          Example: CHECK(ptr != nullptr) << "Null pointer!";
 */
#define CHECK(conditions)                                                      \
  LIKELY(conditions)                                                           \
  ? (void)0                                                                    \
  : dlc_inf::FMessageVoidify() &                                               \
          dlc_inf::FMessageLogger(__FILE__, __LINE__).stream()                 \
              << "Check failed: " #conditions " "


/**
 * @brief Check equality of two values
 * @param val1 First value to compare
 * @param val2 Second value to compare
 * @details If val1 != val2, logs both values and aborts.
 *          Example: CHECK_EQ(a, b) << "a and b should be equal";
 */
#define CHECK_EQ(val1, val2)                                                   \
  LIKELY((val1) == (val2))                                                     \
  ? (void)0                                                                    \
  : dlc_inf::FMessageVoidify() &                                               \
          dlc_inf::FMessageLogger(__FILE__, __LINE__).stream()                 \
              << "Check failed: " #val1 " == " #val2 " (" << (val1) << " vs "  \
              << (val2) << ") "


/**
 * @brief Check that val1 is less than val2
 * @param val1 First value (should be smaller)
 * @param val2 Second value (should be larger)
 * @details If val1 >= val2, logs both values and aborts.
 *          Example: CHECK_LT(index, size) << "Index out of bounds";
 */
#define CHECK_LT(val1, val2)                                                   \
  LIKELY((val1) < (val2))                                                      \
  ? (void)0                                                                    \
  : dlc_inf::FMessageVoidify() &                                               \
          dlc_inf::FMessageLogger(__FILE__, __LINE__).stream()                 \
              << "Check failed: " #val1 " < " #val2 " (" << (val1) << " vs "   \
              << (val2) << ") "


/**
 * @brief Check that val1 is less than or equal to val2
 * @param val1 First value
 * @param val2 Second value
 * @details If val1 > val2, logs both values and aborts.
 *          Example: CHECK_LE(used, capacity) << "Capacity exceeded";
 */
#define CHECK_LE(val1, val2)                                                   \
  LIKELY((val1) <= (val2))                                                     \
  ? (void)0                                                                    \
  : dlc_inf::FMessageVoidify() &                                               \
          dlc_inf::FMessageLogger(__FILE__, __LINE__).stream()                 \
              << "Check failed: " #val1 " <= " #val2 " (" << (val1) << " vs "  \
              << (val2) << ") "


/**
 * @brief Check that val1 is greater than or equal to val2
 * @param val1 First value
 * @param val2 Second value
 * @details If val1 < val2, logs both values and aborts.
 *          Example: CHECK_GE(size, min_size) << "Size too small";
 */
#define CHECK_GE(val1, val2)                                                   \
  LIKELY((val1) >= (val2))                                                     \
  ? (void)0                                                                    \
  : dlc_inf::FMessageVoidify() &                                               \
          dlc_inf::FMessageLogger(__FILE__, __LINE__).stream()                 \
              << "Check failed: " #val1 " >= " #val2 " (" << (val1) << " vs "  \
              << (val2) << ") "

#endif
