#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <ostream>
#include <poputil/TileMapping.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensorlocation.hpp"
#include "popart/tensors.hpp"
#include "popart/util.hpp"

namespace CustomOperators {
const popart::OperatorIdentifier NewIpuCopyId = {"custom.ops", "NewIpuCopy", 1};
} // namespace CustomOperators

class NewIpuCopyOp;
class NewIpuCopyOpx;

using DestIpuMap = std::map<popart::TensorId, popart::VGraphId>;
using DestTensorMap = std::map<popart::VGraphId, std::vector<popart::TensorId>>;

class NewIpuCopyOp : public popart::Op {
public:
  NewIpuCopyOp(const popart::OperatorIdentifier &_opid,
               popart::VGraphId _sourceIpu,
               const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_), sourceIpu(_sourceIpu) {}

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<NewIpuCopyOp>(*this);
    // Reset source info
    // clone->setDestIpus({});
    // clone->setDestTensors({});
    // return std::unique_ptr<Op>(clone);
  }

  void setup() final {
    for (auto &idx_tensor : input->tensorMap()) {
      auto idx = idx_tensor.first;
      outInfo(idx) = inInfo(idx);
    }
  }

  void setDestIpus(const DestIpuMap destIpus_) { destIpus = destIpus_; }

  void setDestTensors(const DestTensorMap destTensors_) {
    destTensors = destTensors_;
  }

  popart::VGraphId getDestIpu(const popart::TensorId &tenId) const {
    return destIpus.at(tenId);
  }

  void connectOutTensor(popart::OutIndex outIndex, popart::TensorId tenId,
                        popart::VGraphId destIpu) {
	  std::cout << "Here is connectOutTensor " <<std::endl;
	  std::cout << "outIndex = " << outIndex <<std::endl;
	  std::cout << "tenId = " << tenId <<std::endl;
     	  destIpus.insert({tenId, destIpu});
    if (destTensors.find(destIpu) == destTensors.end()) {
      destTensors.insert({destIpu, {tenId}});
    } else {
      std::vector<popart::TensorId> &tensorIds = destTensors.at(destIpu);
      tensorIds.push_back(tenId);
    }
    Op::connectOutTensor(outIndex, tenId);
  }

  void appendAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendAttributes(os);
    std::set<int64_t> ipus;
    for (auto &destIpu : destIpus) {
      ipus.insert(destIpu.second);
    }
    os.appendAttribute("__sourceIpu", sourceIpu);
    os.appendAttribute("__destIpus", popart::logging::format("{}", ipus));
  }

  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendOutlineAttributes(os);
  }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  bool requiresRandomSeed() const override { return false; }

  popart::VGraphIdAndTileSet
  getIntrospectionInVirtualGraphId(popart::InIndex index,
                                   std::set<popart::OpId> &visited) const {
    return {sourceIpu, settings.tileSet};
  }

  popart::VGraphIdAndTileSet
  getIntrospectionOutVirtualGraphId(popart::OutIndex index,
                                    std::set<popart::OpId> &visited) const {
    return {destIpus.at(outId(index)), settings.tileSet};
  }

private:
  DestIpuMap destIpus;
  DestTensorMap destTensors;
  popart::VGraphId sourceIpu;
};

namespace {
using popart::DataType;
using popart::OpDefinition;

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition
    newIpuCopyOpDef({OpDefinition::Inputs({{"input", T}}),
                     OpDefinition::Outputs({{"output", T}}),
                     OpDefinition::Attributes({{"__sourceIpu", {"*"}}})});

static popart::OpCreator<NewIpuCopyOp> newIpuCopyOpCreator(
    popart::OpDefinitions({{CustomOperators::NewIpuCopyId, newIpuCopyOpDef}}),
    [](const popart::OpCreatorInfo &info) {
      float __sourceIpu = info.attributes.getAttribute<popart::Attributes::Int>(
          "__sourceIpu", -1);
      return std::make_unique<NewIpuCopyOp>(info.opid, __sourceIpu,
                                            info.settings);
    },
    true);
} // namespace

// namespace pe = popops::expr;

class NewIpuCopyOpx : public popart::popx::Opx {
public:
  NewIpuCopyOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<NewIpuCopyOp>(op, {CustomOperators::NewIpuCopyId});
  }

  void grow(poplar::program::Sequence &prog) const final {
 NewIpuCopyOp &op = getOp<NewIpuCopyOp>();

	    for (auto &idx_tensor : op.output->tensorMap()) {
    auto idx = idx_tensor.first;
    // Need to get the non virtual graph, so cannot use Opx::graph()
    auto t = poputil::copyToIpu(dv_p->lowering().graph(),
                                getInTensor(idx),
                                prog,
                                static_cast<int>(op.getDestIpu(op.output->tensor(idx)->id)),
                                debugContext("newIpuCopy"));
    setOutTensor(idx, t);
  }

  }

};

static popart::popx::OpxCreator<NewIpuCopyOpx>
    NewIpuCopyOpxCreator({CustomOperators::NewIpuCopyId});

