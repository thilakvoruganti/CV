import FeatureCard from "../components/FeatureCard";
import "./HomePage.css";

const features = [
  {
    title: "Measure Lab",
    description:
      "Pick two points on a calibrated capture and convert pixel distances into centimeter-accurate readings with annotated overlays.",
    image:
      "https://images.unsplash.com/photo-1434030216411-0b793f4b4173?auto=format&fit=crop&w=1200&q=80",
    ctaLabel: "Open Measurement",
    to: "/measure",
  },
  {
    title: "Image Stitching",
    description:
      "Fuse a sequence of overlapping photos into a seamless panorama with feature matching, homography alignment, and feather blending.",
    image:
      "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1200&q=80",
    ctaLabel: "Launch Stitcher",
    to: "/stitch",
  },
  {
    title: "SIFT Feature Lab",
    description:
      "Experiment with handcrafted feature detectors, compare against OpenCV baselines, and visualize robust matches.",
    image:
      "https://images.unsplash.com/photo-1520607162513-77705c0f0d4a?auto=format&fit=crop&w=1200&q=80",
    ctaLabel: "Compare detectors",
    to: "/sift",
  },
  {
    title: "Module 2 · Template Lab",
    description:
      "Detect objects via correlation-based template matching, visualize Fourier blur recovery, and blur hits from your template database.",
    image:
      "https://images.unsplash.com/photo-1487412720507-e7ab37603c6f?auto=format&fit=crop&w=1200&q=80",
    ctaLabel: "Open Module 2",
    to: "/module2",
  },
  {
    title: "Module 3 · Edge Lab",
    description:
      "Run gradient diagnostics, edge/corner detectors, and boundary extraction on your measurement dataset—directly in the browser.",
    image:
      "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1200&q=80",
    ctaLabel: "Open Module 3",
    to: "/module3",
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
