import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { AppStateProvider } from "./context/AppStateContext";
import { useAuth } from "./hooks/useAuth";
import LoadingScreen from "./components/common/LoadingScreen/LoadingScreen";
import WelcomeScreen from "./components/onboarding/WelcomeScreen/WelcomeScreen";
import TopicSelector from "./components/onboarding/TopicSelector/TopicSelector";
import AuthPage from "./components/auth/AuthPage/AuthPage";
import MainInterface from "./components/main/MainInterface/MainInterface";

function RequireAuth({ children }) {
  const { user, authChecked } = useAuth();
  const location = useLocation();

  if (!authChecked) return <LoadingScreen label="Checking your session…" />;
  if (!user) return <Navigate to="/auth" replace state={{ from: location }} />;
  return children;
}

function RedirectIfAuthed({ children }) {
  const { user, authChecked } = useAuth();
  if (!authChecked) return <LoadingScreen label="Loading…" />;
  if (user) return <Navigate to="/app" replace />;
  return children;
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<WelcomeScreen />} />
      <Route path="/topics" element={<TopicSelector />} />
      <Route
        path="/auth"
        element={
          <RedirectIfAuthed>
            <AuthPage />
          </RedirectIfAuthed>
        }
      />
      <Route
        path="/app"
        element={
          <RequireAuth>
            <MainInterface />
          </RequireAuth>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <AppStateProvider>
      <AppRoutes />
    </AppStateProvider>
  );
}
