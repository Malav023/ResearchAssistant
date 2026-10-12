import styles from "./BentoCard.module.css";

export default function BentoCard({ as: Tag = "div", className = "", interactive = false, ...props }) {
  const classes = [styles.card, interactive ? styles.interactive : "", className].filter(Boolean).join(" ");
  return <Tag className={classes} {...props} />;
}
