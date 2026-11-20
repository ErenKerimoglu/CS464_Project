"""Small CNN trained from scratch. Issue #6 (owner: Berkay).

4 blocks of (conv3×3-BN-ReLU)×2 + maxpool, widths 32→64→128→256, GAP, dropout 0.3, linear head (~1–2 M params).
"""
