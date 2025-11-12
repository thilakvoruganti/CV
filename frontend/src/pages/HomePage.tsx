import FeatureCard from "../components/FeatureCard";
import "./HomePage.css";

const features = [
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
    title: "Mobile vs DSLR Alignment (Coming Soon)",
    description:
      "Blend captures across devices, compensate for focal discrepancies, and export immersive scenes for AR pipelines.",
    image:
      "https://images.unsplash.com/photo-1500534314209-a25ddb2bd429?auto=format&fit=crop&w=1200&q=80",
    ctaLabel: "Stay tuned",
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
