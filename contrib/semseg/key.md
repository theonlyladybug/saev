# Dimension Key

Throughout the code, variables are annotated with shape suffixes, as [recommended by Noam Shazeer](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd).

The key for these suffixes:

* B: batch size
* W: width in patches (typically 14 or 16)
* H: height in patches (typically 14 or 16)
* D: ViT activation dimension (typically 768 or 1024)
* S: SAE latent dimension (768 x 16, etc)
* L: Number of latents being manipulated at once (typically 1-5 at a time)
* C: Number of classes in ADE20K (151)
