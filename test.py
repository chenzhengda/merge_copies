import numpy as np
import popart

bps = 1

batch_size = 1
seq_len = 256

vocab_size = 50272
n_embd = 1600

import ctypes
ctypes.cdll.LoadLibrary("./custom_library.so") 

builder = popart.Builder()

# Build a simple graph
input_ids = builder.addInputTensor(popart.TensorInfo("INT32", [batch_size, seq_len]))

data_embd_table = np.random.randn(vocab_size, n_embd).astype(np.float32)
data_embd_table_0, data_embd_table_1 = np.split(data_embd_table, 2, 0)

with builder.virtualGraph(0):
    embd_table_0 = builder.addInitializedInputTensor(data_embd_table_0, "embedding_table_0")
    input_ids_0 = builder.aiOnnx.clip([input_ids], min=0, max=vocab_size // 2 - 1)
    inputs_embeds_0 = builder.aiOnnx.gather([embd_table_0, input_ids_0])
    

with builder.virtualGraph(1):
    embd_table_1 = builder.addInitializedInputTensor(data_embd_table_1, "embedding_table_1")
    input_ids_1 = builder.aiOnnx.clip([input_ids], min=vocab_size // 2, max=vocab_size)
    input_ids_1 = builder.aiOnnx.sub([input_ids_1, builder.aiOnnx.constant(np.array([vocab_size // 2], dtype=np.int32))])
    inputs_embeds_1 = builder.aiOnnx.gather([embd_table_1, input_ids_1])
    inputs_embeds = builder.aiOnnx.add([inputs_embeds_0, inputs_embeds_1]) # [bs, seq_len, n_embd]


with builder.virtualGraph(2):
    # [bs, seq_len, n_embd] * [n_embd, 1] = [bs, seq_len, 1]
    data_block0_weight = np.random.randn(n_embd, 1).astype(np.float32)
    block0_weight = builder.addInitializedInputTensor(data_block0_weight, "block0_weight")
    block0_act = builder.aiOnnx.matmul([inputs_embeds, block0_weight])
    # transpose [bs, seq_len, 1] to [bs, 1, seq_len]
    block0_act = builder.aiOnnx.transpose([block0_act], [0,2,1])

with builder.virtualGraph(3):
    # [bs, 1, seq_len] * [seq_len, n_embd] = [bs, 1, n_embd]
    data_block1_weight = np.random.randn(seq_len, n_embd).astype(np.float32)
    block1_weight = builder.addInitializedInputTensor(data_block1_weight, "block1_weight")
    block1_act = builder.aiOnnx.matmul([block0_act, block1_weight])
    
    # [bs, 1, n_embd] to [bs, n_embed]
    hidden_states = builder.aiOnnx.squeeze([block1_act], [1])
    
    # transpose [bs, n_embd] to [n_embed, bs]
    hidden_states = builder.aiOnnx.transpose([hidden_states], [1, 0])

with builder.virtualGraph(0):
    # [vocab_size/2, n_embd] * [n_embed, bs] = [vocab_size/2, bs]
    logits0 = builder.aiOnnx.matmul([embd_table_0, hidden_states])
    
    k0 = builder.aiOnnx.constant(np.array(4).astype(np.int64), 'k')
    next_token_prob0, next_token0 =builder.aiOnnx.topk([logits0, k0], axis=0) # next_token0 = [4, bs]

with builder.virtualGraph(1):
    logits1 = builder.aiOnnx.matmul([embd_table_1, hidden_states])
    
    k1 = builder.aiOnnx.constant(np.array(4).astype(np.int64), 'k')
    next_token_prob1, next_token1 =builder.aiOnnx.topk([logits1, k1], axis=0)  # next_token1 = [4, bs]
    
    next_token = builder.aiOnnx.concat([next_token0, next_token1], 0) # next_token = [8, bs]
    next_token = builder.aiOnnx.argmax([next_token], 0) # next_token = [1, bs]
    o = next_token


builder.addOutputTensor(o)

# Get the ONNX protobuf from the builder to pass to the Session
proto = builder.getModelProto()

# Create a runtime environment 
dataFlow = popart.DataFlow(1, {o : popart.AnchorReturnType("ALL")})
device = popart.DeviceManager().acquireAvailableDevice(4)

opts = popart.SessionOptions()
opts.virtualGraphMode = popart.VirtualGraphMode.Manual
opts.syntheticDataMode = popart.SyntheticDataMode.RandomNormal
opts.experimentalSettings.customTransformApplierSettings = {
    "PreAlias": ["IrSerialise"]
}
# Create the session from the graph, data feed and device information
session = popart.InferenceSession(fnModel=proto,
                            dataFlow=dataFlow,
                            userOptions=opts,
                            deviceInfo=device)

# Compile graph and load graph to IPU
session.prepareDevice()

# Create input data
data_input_ids = np.random.randint(0, vocab_size, [bps, batch_size, seq_len]).astype(np.int32)

# Create PyStepIO
anchors = session.initAnchorArrays()
inputs = {input_ids: data_input_ids}
stepio = popart.PyStepIO(inputs, anchors)

# Run session and retrieve result
session.run(stepio)
result = anchors[o]
print(result)

