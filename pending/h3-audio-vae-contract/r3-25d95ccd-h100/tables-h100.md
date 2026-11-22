<!-- bi -->
| path | dir | 0.9 s, T=37 | 5.0 s, T=202 | 10.0 s, T=402 | 20.0 s, T=802 |
|---|---|---|---|---|---|
| Diffusers | encode | 3.3e-03 | 4.0e-03 | 2.7e-02 | 2.7e-02 |
| Diffusers | decode | 6.4e-04 | 4.1e-02 | 2.5e-02 | 7.0e-03 |
| vLLM-Omni | encode | = 0 | 1.1e-06 | 1.4e-06 | 4.1e-06 |
| vLLM-Omni | decode | 6.4e-04 | 4.1e-02 | 2.5e-02 | 7.0e-03 |
| ours | encode | = 0 | = 0 | = 0 | = 0 |
| ours | decode | = 0 | = 0 | = 0 | = 0 |

Max abs diff. `= 0` is a max abs difference of exactly zero, drawn as `= 0` on the chart (byte equality is the sha256 comparison in the JSON). Source: `h3-audio-vae-h100.json`, commit 25d95ccd, NVIDIA H100.

<!-- accuracy -->
| path | dir | 0.9 s, T=37 | 5.0 s, T=202 | 10.0 s, T=402 | 20.0 s, T=802 |
|---|---|---|---|---|---|
| Diffusers | encode | 7.5e-03 | 1.7e-02 | 3.0e-02 | 2.4e-02 |
| Diffusers | decode | 5.7e-03 | 1.0e-02 | 3.0e-02 | 3.3e-02 |
| vLLM-Omni | encode | 6.6e-06 | 1.6e-05 | 1.0e-04 | 1.0e-04 |
| vLLM-Omni | decode | 5.7e-03 | 1.0e-02 | 3.0e-02 | 3.3e-02 |
| ours | encode | 6.6e-06 | 1.6e-05 | 1.0e-04 | 1.0e-04 |
| ours | decode | 3.1e-05 | 2.0e-05 | 8.8e-05 | 5.6e-05 |

Max abs error. `= 0` is a max abs difference of exactly zero, drawn as `= 0` on the chart (byte equality is the sha256 comparison in the JSON). Source: `h3-audio-vae-h100.json`, commit 25d95ccd, NVIDIA H100.

<!-- coldstart -->
| path | encode | decode |
|---|---|---|
| Diffusers | = 0 | = 0 |
| vLLM-Omni | = 0 | = 0 |
| ours | = 0 | = 0 |

Max abs diff. `= 0` is a max abs difference of exactly zero, drawn as `= 0` on the chart (byte equality is the sha256 comparison in the JSON). Source: `h3-audio-vae-h100.json`, commit 25d95ccd, NVIDIA H100.

<!-- parity -->
| path | encode bitwise = ours | decode bitwise = ours |
|---|---|---|
| Diffusers | 0/4 | 0/4 |
| vLLM-Omni | 3/4 | 0/4 |

Cases bitwise equal to the strict path. Source: `h3-audio-vae-h100.json`, commit 25d95ccd, NVIDIA H100.

<!-- latency -->
| path | dir | 0.9 s, T=37 | 5.0 s, T=202 | 10.0 s, T=402 | 20.0 s, T=802 |
|---|---|---|---|---|---|
| Diffusers | encode | 7.1 | 19.3 | 35.8 | 68.2 |
| Diffusers | decode | 45.5 | 48.5 | 93.5 | 182.0 |
| vLLM-Omni | encode | 57.9 | 65.4 | 90.6 | 146.6 |
| vLLM-Omni | decode | 48.3 | 50.7 | 92.5 | 180.4 |
| ours | encode | 63.8 | 63.6 | 86.6 | 143.0 |
| ours | decode | 84.5 | 84.8 | 120.8 | 248.4 |

Median ms, rounded to 0.1 ms. Source: `h3-audio-vae-h100.json`, commit 25d95ccd, NVIDIA H100.
