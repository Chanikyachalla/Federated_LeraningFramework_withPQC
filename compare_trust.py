import json

d11 = json.load(open('results/Exp11_Data_Poisoning_+_PQC_+_Loss-Based_Trust_metrics.json'))

# The loss-based check uses TRUST_VALIDATION_BATCHES=5 batches
# Let's check: does the evaluation even see a difference?
# If the poisoned update barely changes the loss on 5 test batches,
# loss_raw stays at ~1.0 and falls through to the same score as cosine

# Let's check by looking at the test_loss progression
print("Test loss progression (Exp11 Loss Trust):")
for i in range(0, 50, 5):
    print(f"  Round {i+1}: loss={d11['test_loss'][i]:.4f}, acc={d11['test_accuracy'][i]:.2f}%")

print()
# Let's calculate: what would the loss difference be for a single client?
# base_loss ~ 0.32 (from summary)
# If client_loss ~ 0.33 after applying ONE client's update to a model with 200K params
# relative_delta = (0.33 - 0.32) / 0.32 = 0.031
# loss_raw = exp(-10 * 0.031) = exp(-0.31) = 0.73
# With EMA: new = 0.8*1.0 + 0.2*0.73 = 0.946
# That's barely a penalty!

# But let's check if the loss-based scores are actually being computed differently
# by looking at intermediate rounds where cosine and loss MIGHT differ

d10 = json.load(open('results/Exp10_Data_Poisoning_+_PQC_+_Cosine-Based_Trust_metrics.json'))

print("Comparing trust at round 6 (first round after warm-up):")
ts10_r6 = d10['trust_scores'][5] if len(d10['trust_scores']) > 5 else {}
ts11_r6 = d11['trust_scores'][5] if len(d11['trust_scores']) > 5 else {}

identical = True
for cid in range(10):
    c = float(ts10_r6.get(str(cid), 0))
    l = float(ts11_r6.get(str(cid), 0))
    diff = abs(c - l)
    if diff > 0.0001:
        identical = False
    print(f"  Client {cid}: cosine={c:.6f}, loss={l:.6f}, diff={diff:.8f}")

print()
if identical:
    print("IDENTICAL at round 6 too!")
    print("This means loss evaluation is either not producing different scores,")
    print("or the loss signal is so small it gets drowned out by EMA.")
else:
    print("DIFFERENT at round 6 - the scores converge over time")
