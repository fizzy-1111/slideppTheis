
## Our Contributions
- Investigate synergistic effects of combining multiple loss functions including different GAN loss functions, Feature matching and L1 loss in training HR-VITON.
- Develop complete application pipeline - users input human image and clothing image, system generates final try-on image. The application handles all the processed with user basic inputs, unlike the original which uses pre-prepared data from the dataset.

## Challenges when doing the thesis
- Mode Collapse - the generator collapses to a limited variety of samples, lacking diversity.
- Vanishing Gradients - the discriminator gets too good, causing gradients to vanish and network training to destabilize.
- Non-convergence - the generator and discriminator fail to improve together, resulting in oscillations or plateaus.

## Task that took the most time
Finding Nash Equilibrium: GANs involve finding a Nash equilibrium between the generator and discriminator networks. This balancing act requires extensive hyperparameter tuning and training over many epochs.

## Meaning of each of the loss function
**L1 loss:**
- In GANs, the generator is trained to fool the discriminator and generate realistic looking images.
- However, without any other losses, the generator has no incentive to actually reconstruct the original image faithfully.
- Adding L1 loss between the generated and real images acts as a "reconstruction loss".
- It compares the generated and real images pixel-wise and penalizes large differences.
- Minimizing the L1 reconstruction loss forces the generator to produce images that match the originals as closely as possible.

**Feature matching loss:**
- It extracts features from various layers of the discriminator.
- The loss then calculates the distance between expected features on real images and generated images.
- Reducing this feature distance makes the generator produce outputs that match the statistics of real images.
- This provides a stronger training signal compared to just matching the final discriminator output.

**VGG loss:**
- VGG loss uses a pre-trained VGG neural network to extract feature representations from generated and real images.
- It compares these high-level feature representations rather than pixel values directly.
- This loss captures perceptual and semantic differences between images better than pixel-level losses.

**Hinge loss:**
- Hinge Loss encourages the Discriminator to produce larger differences between its outputs for real and fake samples.
- This can lead to more stable convergence during training.
- By encouraging a clear separation between real and fake samples, Hinge Loss can help prevent mode collapse and promote diversity in the generated samples.

**Least square GAN:**
- L2 loss may help mitigate mode collapse, where the generator produces a limited variety of samples, by providing stronger gradients to the generator.
- It also provides non-saturating gradients, which can help avoid issues like vanishing gradients.

## 3 problems when training GAN - How to fix
**Non-convergence issues:**
- Normally, the generator only tries to fool the discriminator's final real/fake prediction.
- With feature matching, the generator also tries to match the internal features inside the discriminator.
- This gives the generator more clues on how to improve. It has an additional guide.
- By matching the features, the generator learns to produce outputs more similar to real data.
- This helps prevent the generator and discriminator from plateauing and not converging properly.

**Mode collapse problem:**
- Mode collapse is when the generator produces only a limited variety of samples. It gets stuck only generating very similar outputs.
- L1 loss compares the generator's images with real images pixel-by-pixel. It measures how different they are directly.
- By adding L1 loss to the GAN training, the generator can't just fool the discriminator.
- It also has to try and match the pixel values of real images closely.
- This forces the generator to produce varied images that match real data, avoiding mode collapse.

**Vanishing gradients:**
- Vanishing gradients happen when the discriminator gets very good at telling real vs fake.
- This makes the gradients for the generator tiny and useless. Training stagnates.
- LSGAN changes the task to regression instead of binary classification.
- The discriminator tries to output the pixel values of the image rather than just 0 or 1.
- Even if the discriminator gets very good at this, the generator can still get meaningful gradients. Because the pixel outputs are continuous values, not discrete 0/1.
- So the gradients don't suddenly disappear or saturate when the discriminator gets very confident.

## The reasons we choose HR-VITON as the baseline
**Pros:**
- It focuses on high-resolution virtual try-on, generating more detailed and realistic synthesized images compared to previous models like VITON.
- The unified try-on condition generator elegantly combines warping and segmentation map generation. This eliminates misalignment artifacts.
- It incorporates human region parsing and dense pose estimation for better accuracy in alignment.
- The model handles occlusions well by using body part information to remove occluded regions. This reduces pixel-squeezing artifacts.
- Extensive evaluations demonstrate HR-VITON's superior performance over baselines in terms of image quality and realism.

**Cons:**
- The performance relies heavily on the quality of the human parsing and dense pose estimation. Any errors propagate through the pipeline.
- Training the model is computationally intensive due to the need for high resolution inputs and outputs.
- The architectures and objective functions, while effective, are quite complex with many components.

## Architecture of our system
**Pre-processing Module:**
- Takes in a human image and clothing image as input.
- Generates intermediate outputs like clothing mask, dense pose, clothing-agnostic segmentation map and image.

**Try-on Condition Module (GAN):**
- Takes preprocessed outputs and generates warped clothing image and segmentation map.
- Consists of generator (encoder-decoder structure) and discriminator (multi-scale).
- Handles occlusion and misalignment using condition aligning.

**Try-on Image Module (GAN):**
- Takes clothing-agnostic image, warped clothing image, pose map and segmentation map as input.
- Generates final try on image.
The paper mostly use the architecture just like other viton model, which contain a condition module to produce condition for the generator module.

## The difference between our thesis and the baseline
- Incorporate L1 loss in try-on image generator module for training.
- Add other GAN loss functions such as Least Square and Cross-Entropy.
- Establish complete application pipeline:
  - Accept person and clothing images as input.
  - Generate final try-on output image.

## Reasons for choosing the try-on Image module to experiment
- The loss function for the try-on generator already combines multiple losses and works well. It handles common GAN training problems.
- But the try-on image loss can be improved further by adding L1 reconstruction loss.
- L1 loss compares generated and real images pixel-wise. This can enhance image quality.
- After incorporating L1 loss, we will experiment with different loss combinations to find the optimal balance for the try-on image module.
- Testing permutations of L1 with other losses like GAN, FM, perceptual etc. will reveal the best performing blend.
- The goal is to leverage L1's reconstruction properties while retaining the benefits of the existing losses.