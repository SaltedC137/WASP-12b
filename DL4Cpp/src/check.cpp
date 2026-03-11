/**
 * @author Aska Lyn
 * @brief Implementation of logging and assertion utilities
 * @date 2026-03-04 11:10:27
 */

#include "check.hpp"

namespace dlc_inf {

FMessageLogger::FMessageLogger(const char* file, int line) {
  stream_ << file << ":" << line << "] ";
}

FMessageLogger::~FMessageLogger() {
  std::cerr << stream_.str() << std::endl;
  std::abort();
}

std::ostream& FMessageLogger::stream() {
  return stream_;
}

void FMessageVoidify::operator&(std::ostream& stream) {
  (void)stream;
}

} // namespace dlc_inf
