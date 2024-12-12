#pragma once

#include <Eigen/Core>
#include <memory_resource>

namespace aligator {
namespace {
using Eigen::DenseBase;
}

using byte_t = unsigned char;
using polymorphic_allocator = std::pmr::polymorphic_allocator<byte_t>;

/// \brief A subclass of Eigen::Map which actually manages memory from a pool.
template <typename _Scalar, int Rows = Eigen::Dynamic,
          int Cols = Eigen::Dynamic>
class EigenPmr
    : public Eigen::Map<Eigen::Matrix<_Scalar, Rows, Cols>, Eigen::AlignedMax> {
public:
  using Index = Eigen::Index;
  using Traits = Eigen::internal::traits<EigenPmr>;
  using PlainObjectType = Eigen::Matrix<_Scalar, Rows, Cols>;
  using Base = Eigen::Map<PlainObjectType, Eigen::AlignedMax>;
  EIGEN_DENSE_PUBLIC_INTERFACE(EigenPmr)

  // allocator of byte_t, to avoid having to deal with Scalar's alignment...
  using allocator_type = polymorphic_allocator;

  static constexpr size_t AlignmentRequirement = Traits::Alignment;

  void *aligned_alloc_impl(allocator_type &alloc, size_t size_bytes,
                           size_t align) {
    size_t total_size_bytes = size_bytes + align;
    m_orig_ptr = alloc.allocate(total_size_bytes);
    if (!m_orig_ptr) {
      throw std::bad_alloc();
    }

    size_t space = total_size_bytes;
    void *ptr_copy = m_orig_ptr;
    void *align_ptr = std::align(align, size_bytes, ptr_copy, space);
    if (!align_ptr) {
      alloc.deallocate(m_orig_ptr, total_size_bytes);
      throw std::bad_alloc();
    }
    return (byte_t *)align_ptr;
  }

  Scalar *aligned_alloc(allocator_type &alloc, size_t size,
                        size_t align = AlignmentRequirement) {
    return reinterpret_cast<Scalar *>(
        aligned_alloc_impl(alloc, sizeof(Scalar) * size, align));
  }

  explicit EigenPmr(const allocator_type &alloc = {})
      : Base(NULL), m_allocator(alloc) {
    PlainObjectType::Base::_check_template_params();
  }

  explicit EigenPmr(Index size, allocator_type alloc = {})
      : Base(aligned_alloc(alloc, size), size), m_allocator(alloc) {
    PlainObjectType::Base::_check_template_params();
  }

  EigenPmr(Index rows, Index cols, allocator_type alloc = {})
      : Base(aligned_alloc(alloc, rows * cols), rows, cols),
        m_allocator(alloc) {
    PlainObjectType::Base::_check_template_params();
  }

  EigenPmr(const EigenPmr &other)
      : EigenPmr(other.rows(), other.cols(), other.get_allocator()) {
    memcpy(this->m_data, other.data(), other.size() * sizeof(Scalar));
  }

  EigenPmr(EigenPmr &&other) : Base(other.m_data, other.m_rows, other.m_cols) {
    // use base map class ctor
    this->m_data = other.m_data;
    this->m_orig_ptr = other.m_orig_ptr;
    other.m_data = NULL;
    other.m_orig_ptr = NULL;
  }

  ~EigenPmr() {
    if (this->m_data) {
      auto dealloc_bytes = this->size() * sizeof(Scalar) + AlignmentRequirement;
      m_allocator.deallocate(m_orig_ptr, dealloc_bytes);
      this->m_data = NULL;
      this->m_orig_ptr = NULL;
    }
  }

  allocator_type get_allocator() const noexcept { return m_allocator; }

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(EigenPmr);

protected:
  allocator_type m_allocator;
  byte_t *m_orig_ptr;
};

} // namespace aligator

namespace Eigen::internal {
template <typename Scalar, int Rows, int Cols>
struct traits<aligator::EigenPmr<Scalar, Rows, Cols>>
    : public traits<typename aligator::EigenPmr<Scalar, Rows, Cols>::Base> {
  typedef typename aligator::EigenPmr<Scalar, Rows, Cols>::Base Base;
  typedef traits<Base> TraitsBase;

private:
  enum { Options }; // Expressions don't have Options
};

} // namespace Eigen::internal
