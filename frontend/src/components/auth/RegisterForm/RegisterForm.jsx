import { useState } from "react";
import Button from "../../common/Button/Button";
import styles from "./RegisterForm.module.css";

export default function RegisterForm({ onSubmit, loading, error }) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit?.(name, email, password);
  };

  return (
    <form className={styles.form} onSubmit={handleSubmit}>
      <label className={styles.field}>
        <span className={styles.label}>Name</span>
        <input
          type="text"
          required
          value={name}
          onChange={(e) => setName(e.target.value)}
          className={styles.input}
          autoComplete="name"
        />
      </label>

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
          autoComplete="new-password"
        />
        <span className={styles.hint}>At least 8 characters.</span>
      </label>

      {error && <div className={styles.error}>{error}</div>}

      <Button type="submit" size="lg" disabled={loading} className={styles.submit}>
        {loading ? "Creating account…" : "Create account"}
      </Button>
    </form>
  );
}
