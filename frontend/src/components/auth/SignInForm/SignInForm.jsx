import { useState } from "react";
import Button from "../../common/Button/Button";
import styles from "./SignInForm.module.css";

export default function SignInForm({ onSubmit, loading, error }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit?.(email, password);
  };

  return (
    <form className={styles.form} onSubmit={handleSubmit}>
      <label className={styles.field}>
        <span className={styles.label}>Email</span>
        <input
          type="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className={styles.input}
          autoComplete="email"
        />
      </label>

      <label className={styles.field}>
        <span className={styles.label}>Password</span>
        <input
          type="password"
          required
          minLength={8}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className={styles.input}
          autoComplete="current-password"
        />
      </label>

      {error && <div className={styles.error}>{error}</div>}

      <Button type="submit" size="lg" disabled={loading} className={styles.submit}>
        {loading ? "Signing in…" : "Sign in"}
      </Button>
    </form>
  );
}
