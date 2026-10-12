import { useState } from "react";
import { useNavigate } from "react-router-dom";
import BentoCard from "../../common/BentoCard/BentoCard";
import LoadingScreen from "../../common/LoadingScreen/LoadingScreen";
import SignInForm from "../SignInForm/SignInForm";
import RegisterForm from "../RegisterForm/RegisterForm";
import { useAuth } from "../../../hooks/useAuth";
import { useAppState } from "../../../context/AppStateContext";
import { saveTopics } from "../../../api/authApi";
import styles from "./AuthPage.module.css";

export default function AuthPage() {
  const [mode, setMode] = useState("signin"); // signin | register
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [transitioning, setTransitioning] = useState(false);
  const { login, register } = useAuth();
  const { pendingTopics, setPendingTopics } = useAppState();
  const navigate = useNavigate();

  const handleSignIn = async (email, password) => {
    setLoading(true);
    setError(null);
    try {
      await login(email, password);
      setTransitioning(true);
      setTimeout(() => navigate("/app"), 500);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const handleRegister = async (name, email, password) => {
    setLoading(true);
    setError(null);
    try {
      await register(name, email, password);
      if (pendingTopics && pendingTopics.length > 0) {
        try {
          await saveTopics(pendingTopics);
        } catch {
          // topics are cosmetic for v1 — don't block onboarding if this fails
        } finally {
          setPendingTopics([]);
        }
      }
      setTransitioning(true);
      setTimeout(() => navigate("/app"), 500);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  if (transitioning) {
    return <LoadingScreen label="Setting up your workspace…" />;
  }

  return (
    <div className={styles.wrap}>
      <BentoCard className={styles.card}>
        <div className={styles.tabs}>
          <button
            className={`${styles.tab} ${mode === "signin" ? styles.tabActive : ""}`}
            onClick={() => {
              setMode("signin");
              setError(null);
            }}
            type="button"
          >
            Sign in
          </button>
          <button
            className={`${styles.tab} ${mode === "register" ? styles.tabActive : ""}`}
            onClick={() => {
              setMode("register");
              setError(null);
            }}
            type="button"
          >
            Register
          </button>
        </div>

        {mode === "signin" ? (
          <SignInForm onSubmit={handleSignIn} loading={loading} error={error} />
        ) : (
          <RegisterForm onSubmit={handleRegister} loading={loading} error={error} />
        )}
      </BentoCard>
    </div>
  );
}
