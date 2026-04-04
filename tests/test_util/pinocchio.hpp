
#include <optional>
#include <pinocchio/algorithm/contact-info.hpp>

inline auto &
get_baumgarte_corrector_params(pinocchio::RigidConstraintModel &rcm) {
#ifdef ALIGATOR_PINOCCHIO_V4
  return rcm.baumgarte_corrector_parameters();
#else
  return rcm.corrector;
#endif
}

inline void set_baumgarte_gains(pinocchio::RigidConstraintModel &rcm,
                                const double Kp,
                                std::optional<double> Kd_ = std::nullopt) {
  double Kd = Kd_.value_or(Kp);
  auto &corr = get_baumgarte_corrector_params(rcm);
#ifdef ALIGATOR_PINOCCHIO_V4
  corr.Kp = Kp;
  corr.Kd = Kd;
#else
  corr.Kp.array() = Kp;
  corr.Kd.array() = Kd;
#endif
}
