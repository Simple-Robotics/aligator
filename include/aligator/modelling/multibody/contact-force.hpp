/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @brief ContactForceResidual definition.
#pragma once

#include "aligator/core/function-abstract.hpp"

#include "aligator/modelling/multibody/fwd.hpp"
#include "aligator/modelling/dynamics/multibody-constraint-common.hpp"

namespace aligator {
template <typename Scalar> struct ContactForceDataTpl;

struct ContactReference {
  using RigidConstraintModelVector = PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(
      pinocchio::RigidConstraintModel);

  ContactReference(const RigidConstraintModelVector &contact_models,
                   std::size_t p_contact_index)
      : contact_index(p_contact_index) {
    if (contact_index >= contact_models.size()) {
      ALIGATOR_RUNTIME_ERROR(
          fmt::format("Contact index is out of range ({} >= {})", contact_index,
                      contact_models.size()));
    }

    force_index = 0;
    for (std::size_t i = 0; i < contact_index; ++i) {
      const auto &contact = contact_models[i];
      force_index += contact.size();
    }
    force_size = contact_models[contact_index].size();
  }

  std::size_t contact_index;
  Eigen::DenseIndex force_index;
  Eigen::DenseIndex force_size;
};

template <typename _Scalar>
struct ContactForceResidualTpl : StageFunctionTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_STAGE_FUNCTION_TYPEDEFS(Scalar, ContactForceDataTpl);

  using MultibodyConstraintCommon =
      dynamics::MultibodyConstraintCommonTpl<Scalar>;
  using PinocchioData = pinocchio::DataTpl<Scalar>;

  ContactForceResidualTpl(const int ndx, const int nu,
                          const ContactReference &contact_ref)
      : Base(ndx, nu, contact_ref.force_size), contact_reference_(contact_ref) {
  }

  const Vector3s &getReference() const { return reference_; }
  void setReference(const Eigen::Ref<const Vector3s> &ref) { reference_ = ref; }

  const ContactReference &getContactReference() const {
    return contact_reference_;
  }

  void configure(CommonModelBuilderContainer &container) const override {
    container.template get<MultibodyConstraintCommon>().withRunAba(true);
  }

  void evaluate(const ConstVectorRef &, const ConstVectorRef &,
                const ConstVectorRef &, BaseData &data) const override {
    Data &d = static_cast<Data &>(data);
    const auto &contact_data =
        d.multibody_data_->constraint_datas_[contact_reference_.contact_index];
    d.value_ = contact_data.contact_force.toVector().head(
                   contact_reference_.force_size) -
               reference_;
  }

  void computeJacobians(const ConstVectorRef &, const ConstVectorRef &,
                        const ConstVectorRef &, BaseData &data) const override {
    Data &d = static_cast<Data &>(data);
    const Eigen::DenseIndex nv = d.multibody_data_->nv;
    const Eigen::DenseIndex row = contact_reference_.force_index;
    const Eigen::DenseIndex size = contact_reference_.force_size;
    const PinocchioData &pin_data = d.multibody_data_->pin_data_;
    d.Jx_.leftCols(nv) = pin_data.dlambda_dq.middleRows(row, size);
    d.Jx_.rightCols(nv) = pin_data.dlambda_dv.middleRows(row, size);
    d.Ju_.leftCols(nv).noalias() =
        pin_data.dlambda_dtau * d.multibody_data_->actuation_matrix_;
  }

  shared_ptr<BaseData>
  createData(const CommonModelDataContainer &container) const override {
    return std::make_shared<Data>(this, container);
  }

  /// @brief Instantiate a Data object.
  shared_ptr<BaseData> createData() const override {
    ALIGATOR_RUNTIME_ERROR("createdata can't be called without arguments");
  }

protected:
  const ContactReference contact_reference_;
  VectorXs reference_;
};

template <typename _Scalar>
struct ContactForceDataTpl : StageFunctionDataTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_STAGE_FUNCTION_DATA_TYPEDEFS(Scalar, ContactForceResidualTpl);

  using MultibodyConstraintCommon =
      dynamics::MultibodyConstraintCommonTpl<Scalar>;
  using MultibodyConstraintCommonData =
      dynamics::MultibodyConstraintCommonDataTpl<Scalar>;

  ContactForceDataTpl(const Model *model,
                      const CommonModelDataContainer &container)
      : Base(model->ndx1, model->nu, model->ndx2, model->nr),
        multibody_data_(
            container.template getData<MultibodyConstraintCommon>()) {}

  const MultibodyConstraintCommonData *multibody_data_;
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/modelling/multibody/contact-force.txx"
#endif
