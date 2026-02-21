import FeatureCard from "../components/FeatureCard";
import "./HomePage.css";

const moduleImage = (name: string) => `${process.env.PUBLIC_URL}/images/modules/${name}.svg`;

const features = [
  {
    title: "Measurement",
    description:
      "Pick two points on a calibrated capture and convert pixel distances into centimeter-accurate readings with annotated overlays.",
    image: moduleImage("measurement"),
    ctaLabel: "Open Measurement",
    to: "/measure",
  },
  {
    title: "Detection",
    description:
      "Detect objects via correlation-based template matching, visualize Fourier blur recovery, and blur hits from your template database.",
    image: moduleImage("detection"),
    ctaLabel: "Open Detection",
    to: "/module2",
  },
  {
    title: "Edge Lab",
    description:
      "Run gradient diagnostics, edge/corner detectors, and boundary extraction on your measurement dataset—directly in the browser.",
    image: moduleImage("edge-lab"),
    ctaLabel: "Open Edge Lab",
    to: "/module3",
  },
  {
    title: "Tracking",
    description:
      "Launch Shi–Tomasi + Lucas–Kanade tracking, inspect tracked ROIs, and visualize motion paths frame-by-frame using the updated backend.",
    image: moduleImage("tracking"),
    ctaLabel: "Open Tracking",
    to: "/module5",
  },
  {
    title: "Stereo + Pose",
    description:
      "Upload calibrated stereo pairs, lay down correspondences, and recover metric dimensions, diameters, and areas from disparity-driven 3D points.",
    image: moduleImage("stereo-pose"),
    ctaLabel: "Open Stereo + Pose",
    to: "/module7",
  },
  {
    title: "Feature Match",
    description:
      "Experiment with handcrafted feature detectors, compare against OpenCV baselines, and visualize robust matches.",
    image: moduleImage("feature-match"),
    ctaLabel: "Compare features",
    to: "/sift",
  },
  {
    title: "Stitching",
    description:
      "Fuse a sequence of overlapping photos into a seamless panorama with feature matching, homography alignment, and feather blending.",
    image: moduleImage("stitching"),
    ctaLabel: "Launch Stitcher",
    to: "/stitch",
  },
];

export function HomePage() {
  return (
    <section className="home-root">
      <div className="home-hero">
        <p className="home-kicker">Computer Vision Playground</p>
        <h1>Bring pixels together and craft immersive panoramas.</h1>
        <p className="home-subhead">
          Explore practical vision workflows—stitch multi-shot scenes, inspect feature matches, and export ready-to-share composites powered
          by FastAPI + OpenCV.
        </p>
      </div>

      <div className="home-grid">
        {features.map((feature) => (
          <FeatureCard key={feature.title} {...feature} />
        ))}
      </div>
    </section>
  );
}

export default HomePage;
