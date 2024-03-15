#include <boost/test/unit_test.hpp>

#include "aligator/core/common-model-abstract.hpp"
#include "aligator/core/common-model-container.hpp"
#include "aligator/core/common-model-builder-container.hpp"

using CommonModel = aligator::CommonModelTpl<double>;
using CommonModelData = aligator::CommonModelDataTpl<double>;
using CommonModelBuilder = aligator::CommonModelBuilderTpl<double>;
using CommonModelContainer = aligator::CommonModelContainerTpl<double>;
using CommonModelBuilderContainer =
    aligator::CommonModelBuilderContainerTpl<double>;

BOOST_AUTO_TEST_SUITE(common)

/// Contains data to see if evaluate, computeGradients and computeHessians are
/// called
struct MockCommonModelData : CommonModelData {
  bool evaluate_called = false;
  bool evaluate_feature_a = false;
  bool compute_gradients_called = false;
  bool compute_hessians_called = false;

  void reset() {
    evaluate_called = false;
    evaluate_feature_a = false;
    compute_gradients_called = false;
    compute_hessians_called = false;
  }
};

/// Model with a feature to validate the builder options
struct MockCommonModel : CommonModel {
  using BaseData = typename CommonModel::Data;
  using Data = MockCommonModelData;

  bool feature_a;

  /// @brief Evaluate the common model.
  void evaluate(const ConstVectorRef &, const ConstVectorRef &,
                BaseData &base_data) const override {
    auto &data = static_cast<Data &>(base_data);
    data.evaluate_called = true;
    if (feature_a) {
      data.evaluate_feature_a = true;
    }
  }

  /// @brief Compute the common model gradients.
  void computeGradients(const ConstVectorRef &, const ConstVectorRef &,
                        BaseData &base_data) const override {
    auto &data = static_cast<Data &>(base_data);
    data.compute_gradients_called = true;
  }

  /// @brief Compute the common model Hessians.
  void computeHessians(const ConstVectorRef &, const ConstVectorRef &,
                       BaseData &base_data) const override {
    auto &data = static_cast<Data &>(base_data);
    data.compute_hessians_called = true;
  }

  std::shared_ptr<BaseData> createData() const override {
    return std::make_shared<Data>();
  }

  struct Builder : CommonModelBuilder {
    std::shared_ptr<CommonModel> build() const override {
      auto model = std::make_shared<MockCommonModel>();
      model->feature_a = with_feature_a;
      return model;
    }

    Builder &withFeatureA(bool active) {
      with_feature_a = active;
      return *this;
    }

  private:
    bool with_feature_a = false;
  };
};

BOOST_AUTO_TEST_CASE(builder_and_container_one_common) {
  CommonModelBuilderContainer builder_container;
  /// Create and configure bulider
  auto &model_builder = builder_container.get<MockCommonModel>();
  model_builder.withFeatureA(true);

  /// Generate all model and datas
  /// We should have one MockCommonModel and MockCommonModelData
  auto model_container = builder_container.createCommonModelContainer();
  auto data_container = model_container.createData();
  BOOST_REQUIRE_EQUAL(model_container.size(), 1);
  BOOST_REQUIRE_EQUAL(data_container.size(), 1);
  auto *model =
      dynamic_cast<MockCommonModel *>(model_container.at(0).model.get());
  BOOST_REQUIRE(model);
  BOOST_REQUIRE(model->feature_a);
  auto *data =
      dynamic_cast<MockCommonModelData *>(data_container.at(0).data.get());
  BOOST_REQUIRE(data);
  BOOST_REQUIRE(!data->evaluate_called);
  BOOST_REQUIRE(!data->evaluate_feature_a);
  BOOST_REQUIRE(!data->compute_gradients_called);
  BOOST_REQUIRE(!data->compute_hessians_called);

  /// Test get_common_data function
  auto *data_from_type = data_container.getData<MockCommonModel>();
  BOOST_REQUIRE_EQUAL(data_from_type, data);

  /// Test the evaluation
  Eigen::VectorXd x = Eigen::VectorXd::Zero(1);
  Eigen::VectorXd u = Eigen::VectorXd::Zero(1);

  model_container.evaluate(x, u, data_container);
  BOOST_REQUIRE(data->evaluate_called);
  BOOST_REQUIRE(data->evaluate_feature_a);
  BOOST_REQUIRE(!data->compute_gradients_called);
  BOOST_REQUIRE(!data->compute_hessians_called);

  data->reset();
  for (std::size_t i = 0; i < model_container.size(); ++i) {
    model_container[i].model->computeGradients(x, u, *data_container[i].data);
  }
  BOOST_REQUIRE(!data->evaluate_called);
  BOOST_REQUIRE(!data->evaluate_feature_a);
  BOOST_REQUIRE(data->compute_gradients_called);
  BOOST_REQUIRE(!data->compute_hessians_called);

  data->reset();
  for (std::size_t i = 0; i < model_container.size(); ++i) {
    model_container[i].model->computeHessians(x, u, *data_container[i].data);
  }
  BOOST_REQUIRE(!data->evaluate_called);
  BOOST_REQUIRE(!data->evaluate_feature_a);
  BOOST_REQUIRE(!data->compute_gradients_called);
  BOOST_REQUIRE(data->compute_hessians_called);
}

BOOST_AUTO_TEST_CASE(builder_and_container_two_common) {
  CommonModelBuilderContainer builder_container;
  /// Create and configure bulider
  auto &model_builder = builder_container.get<MockCommonModel>();
  model_builder.withFeatureA(true);
  auto model2_builder = std::dynamic_pointer_cast<MockCommonModel::Builder>(
      builder_container.getFromTypeIndexName(
          "MockCommonModel2", std::make_shared<MockCommonModel::Builder>()));
  BOOST_REQUIRE(model2_builder);
  model2_builder->withFeatureA(false);

  /// Generate all model and datas
  /// We should have one MockCommonModel and MockCommonModelData
  auto model_container = builder_container.createCommonModelContainer();
  auto data_container = model_container.createData();
  BOOST_REQUIRE_EQUAL(model_container.size(), 2);
  BOOST_REQUIRE_EQUAL(data_container.size(), 2);

  /// Test get_common_data function
  auto *data = data_container.getData<MockCommonModel>();
  auto data2 = std::dynamic_pointer_cast<MockCommonModel::Data>(
      data_container.getDataFromTypeIndexName("MockCommonModel2"));
  BOOST_REQUIRE(data2);

  /// Test the evaluation
  Eigen::VectorXd x = Eigen::VectorXd::Zero(1);
  Eigen::VectorXd u = Eigen::VectorXd::Zero(1);

  model_container.evaluate(x, u, data_container);
  BOOST_REQUIRE(data->evaluate_called);
  BOOST_REQUIRE(data->evaluate_feature_a);
  BOOST_REQUIRE(!data->compute_gradients_called);
  BOOST_REQUIRE(!data->compute_hessians_called);
  BOOST_REQUIRE(data2->evaluate_called);
  BOOST_REQUIRE(!data2->evaluate_feature_a);
  BOOST_REQUIRE(!data2->compute_gradients_called);
  BOOST_REQUIRE(!data2->compute_hessians_called);

  data->reset();
  data2->reset();
  for (std::size_t i = 0; i < model_container.size(); ++i) {
    model_container[i].model->computeGradients(x, u, *data_container[i].data);
  }
  BOOST_REQUIRE(!data->evaluate_called);
  BOOST_REQUIRE(!data->evaluate_feature_a);
  BOOST_REQUIRE(data->compute_gradients_called);
  BOOST_REQUIRE(!data->compute_hessians_called);
  BOOST_REQUIRE(!data2->evaluate_called);
  BOOST_REQUIRE(!data2->evaluate_feature_a);
  BOOST_REQUIRE(data2->compute_gradients_called);
  BOOST_REQUIRE(!data2->compute_hessians_called);

  data->reset();
  data2->reset();
  for (std::size_t i = 0; i < model_container.size(); ++i) {
    model_container[i].model->computeHessians(x, u, *data_container[i].data);
  }
  BOOST_REQUIRE(!data->evaluate_called);
  BOOST_REQUIRE(!data->evaluate_feature_a);
  BOOST_REQUIRE(!data->compute_gradients_called);
  BOOST_REQUIRE(data->compute_hessians_called);
  BOOST_REQUIRE(!data2->evaluate_called);
  BOOST_REQUIRE(!data2->evaluate_feature_a);
  BOOST_REQUIRE(!data2->compute_gradients_called);
  BOOST_REQUIRE(data2->compute_hessians_called);
}

BOOST_AUTO_TEST_SUITE_END()
