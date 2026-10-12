import styles from "./Button.module.css";

export default function Button({
  variant = "primary", // primary | secondary | ghost
  size = "md", // sm | md | lg
  className = "",
  ...props
}) {
  const classes = [styles.btn, styles[variant], styles[size], className].filter(Boolean).join(" ");
  return <button className={classes} {...props} />;
}
