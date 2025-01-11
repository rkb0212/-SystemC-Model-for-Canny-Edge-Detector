# -SystemC-Model-for-Canny-Edge-Detector
• Developed a SystemC model with a structured DUT module, encapsulating key image-processing stages like
 gaussian-smooth, derivatives-x-y, magnitude-x-y, non-max-supp, and apply-hysteresis into submodules. Added
 submodules for x and y-direction smoothing for scalability and reuse.
 • Conducted profiling using gprof to analyze runtime complexity, optimizing critical stages for better embedded
 system performance.
 • Designed a comprehensive testbench to validate the edge-detection pipeline and integrated timing analysis for
 system-level verification.
