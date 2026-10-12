import { useCallback, useEffect } from "react";
import { fetchMe, login as loginApi, register as registerApi, logout as logoutApi } from "../api/authApi";
import { useAppState } from "../context/AppStateContext";

export function useAuth() {
  const { user, setUser, authChecked, setAuthChecked } = useAppState();

  useEffect(() => {
    let cancelled = false;
    fetchMe()
      .then((u) => {
        if (!cancelled) setUser(u);
      })
      .catch(() => {
        if (!cancelled) setUser(null);
      })
      .finally(() => {
        if (!cancelled) setAuthChecked(true);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const login = useCallback(
    async (email, password) => {
      const u = await loginApi(email, password);
      setUser(u);
      return u;
    },
    [setUser]
  );

  const register = useCallback(
    async (name, email, password) => {
      const u = await registerApi(name, email, password);
      setUser(u);
      return u;
    },
    [setUser]
  );

  const logout = useCallback(async () => {
    await logoutApi();
    setUser(null);
  }, [setUser]);

  return { user, authChecked, login, register, logout };
}
