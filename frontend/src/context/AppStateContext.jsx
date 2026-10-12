import { createContext, useCallback, useContext, useMemo, useState } from "react";

const AppStateContext = createContext(null);

export function AppStateProvider({ children }) {
  const [user, setUser] = useState(null);
  const [authChecked, setAuthChecked] = useState(false);

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activePaper, setActivePaper] = useState(null); // paper detail panel
  const [graphOpen, setGraphOpen] = useState(false);
  const [graphQuery, setGraphQuery] = useState("");
  const [pendingTopics, setPendingTopics] = useState([]); // chosen pre-signup, applied after register

  const toggleSidebar = useCallback(() => setSidebarOpen((v) => !v), []);
  const openPaperDetail = useCallback((paper) => setActivePaper(paper), []);
  const closePaperDetail = useCallback(() => setActivePaper(null), []);
  const openGraph = useCallback((query) => {
    setGraphQuery(query);
    setGraphOpen(true);
  }, []);
  const closeGraph = useCallback(() => setGraphOpen(false), []);

  const value = useMemo(
    () => ({
      user,
      setUser,
      authChecked,
      setAuthChecked,
      sidebarOpen,
      toggleSidebar,
      setSidebarOpen,
      activePaper,
      openPaperDetail,
      closePaperDetail,
      graphOpen,
      graphQuery,
      openGraph,
      closeGraph,
      pendingTopics,
      setPendingTopics,
    }),
    [user, authChecked, sidebarOpen, activePaper, graphOpen, graphQuery, pendingTopics]
  );

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState() {
  const ctx = useContext(AppStateContext);
  if (!ctx) throw new Error("useAppState must be used within AppStateProvider");
  return ctx;
}
