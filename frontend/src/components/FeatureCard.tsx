import { Link } from "react-router-dom";
import "./FeatureCard.css";

export interface FeatureCardProps {
  title: string;
  description: string;
  image: string;
  ctaLabel: string;
  to: string;
}

export function FeatureCard({ title, description, image, ctaLabel, to }: FeatureCardProps) {
  return (
    <article className="feature-card">
      <div className="feature-card__image">
        <img src={image} alt={title} loading="lazy" />
      </div>
      <div className="feature-card__body">
        <h3>{title}</h3>
        <p>{description}</p>
        <Link className="feature-card__cta" to={to}>
          {ctaLabel}
        </Link>
      </div>
    </article>
  );
}

export default FeatureCard;
