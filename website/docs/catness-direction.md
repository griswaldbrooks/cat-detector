---
sidebar_position: 6
title: "The Catness Direction"
---

# The Catness Direction

> *"May the odds be ever in your favor."*
> — Effie Trinket, on the importance of good class separation

![May the catness be ever in your favor](/img/catness-hunger-games.jpg)

In the Hunger Games, every tribute gets thrown into an arena and the odds determine who survives. In our cat detector, every image gets thrown into CLIP's 512-dimensional embedding space and the **cosine similarity** determines who gets classified as a cat. The question is: can we rig the odds?

## The Problem with Centroids

The [few-shot prototype approach](./few-shot-detection.md) represents each class by the **mean embedding** of curated reference images. This works in theory, but in practice the centroids are dangerously close together:

| Pair | Cosine Similarity |
|---|---|
| cat &harr; empty room | 0.978 |
| cat &harr; person | 0.917 |
| cat &harr; litter box | 0.876 |
| empty room &harr; person | 0.928 |

The cat centroid is **97.8% similar** to the empty room centroid. Why? Because a "cat image" from our overhead camera is 90% floor, furniture, and litter robot — with a cat in it. The centroid is dominated by the scene, not the cat.

With softmax temperature &tau;=100, a cosine similarity difference of 0.01 becomes a probability swing of ~63%. So tiny prototype quality issues — one bad image, captures from only one session — cause wild detection swings. The prototype set has to be curated, and when the room changes (new litter box, moved furniture), you have to re-curate.

## The Insight: Subtraction

Instead of asking "does this image look like the average cat image?", ask: **"what's different about this image compared to a non-cat image?"**

```
cat_mean    = average(CLIP_encode(all cat images))
noncat_mean = average(CLIP_encode(all non-cat images))

catness_direction = normalize(cat_mean - noncat_mean)
```

This subtraction cancels out everything shared between cat and non-cat images — the floor, the litter robot, the furniture, the lighting. What survives is the **catness signal**: the dimensions in CLIP's embedding space that change when a cat appears.

For any new image, project onto this direction:

```
catness_score = dot(CLIP_encode(image), catness_direction)
```

Positive = cat-like. Negative = not cat-like. Zero = ambiguous.

## Results

Using test images (57 cat, 12 non-cat):

| Group | Score Range | Threshold @ 0.15 |
|---|---|---|
| Cats (fully visible) | 0.19 to 0.33 | All pass |
| Cats (inside litter robot) | 0.01 to 0.10 | Correctly borderline |
| Litter robot moving | 0.06 | Reject |
| Empty room | -0.06 to -0.08 | Reject |
| Dirty litter box | -0.12 | Reject |
| Person | -0.25 to -0.04 | Reject |

Compare this to the prototype centroid approach, where most overhead cat images scored 20-45% and mixed with non-cat scores. The catness direction produces a **clean gap** between 0.10 and 0.19 where the threshold sits.

### Why it works better

The centroid approach compares a 512-d vector to another 512-d vector where ~466 dimensions are shared scene information. The catness direction **zeros out** those shared dimensions, focusing the comparison on the ~46 dimensions that actually carry cat signal.

| Signal concentration | Dimensions needed |
|---|---|
| 50% of catness signal | 46 / 512 |
| 80% | 136 / 512 |
| 90% | 200 / 512 |
| 99% | 358 / 512 |

Half the discriminative power lives in fewer than 10% of the dimensions.

## Advantages Over Prototypes

| Aspect | Prototype Centroids | Catness Direction |
|---|---|---|
| Background class needed? | Yes (and sensitive to curation) | No |
| Room layout changes? | Must re-curate prototypes | Only affects shared dimensions (canceled out) |
| Number of classes | Must enumerate all negatives | Binary: cat or not-cat |
| Failure mode | "Forgot a class" &rarr; false positive | Threshold too low/high |
| Softmax temperature | &tau;=100 amplifies tiny errors | Not needed (direct threshold) |

## Personness Direction

The same technique works for person detection:

```
person_mean    = average(CLIP_encode(all person images))
nonperson_mean = average(CLIP_encode(all non-person images))

personness_direction = normalize(person_mean - nonperson_mean)
```

This gives two independent scores per image — catness and personness — each with its own threshold. No softmax competition between classes.

## Open Questions

1. **Training data**: The catness direction should be computed from prototype images (training set), not test images. Does the separation hold when computed from the smaller prototype set?
2. **Generalization**: The current test images are from the same camera/room. Does the direction generalize across scene changes?
3. **Threshold stability**: Is 0.15 robust, or does it need tuning per-deployment?
4. **Two cats**: Both cats (light Ragdoll and tabby) contribute to the cat mean. Does one dominate the direction?

## Next Steps

- Compute catness direction from `prototype_images/` only (not test images)
- Validate against test images as held-out data
- If separation holds, implement in Rust as an alternative to softmax classification
- Consider shipping both: catness direction for cat/not-cat, with personness as a secondary filter

## References

1. Snell, J., Swersky, K., & Zemel, R. (2017). [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175). *NeurIPS 2017*.
2. Radford, A., et al. (2021). [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020). *ICML 2021*.
