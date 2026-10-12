import { useEffect, useState } from "react";
import AppShell from "../../layout/AppShell/AppShell";
import Sidebar from "../../layout/Sidebar/Sidebar";
import ChatBar from "../../search/ChatBar/ChatBar";
import ResultsList from "../../search/ResultsList/ResultsList";
import PaperDetailPanel from "../../search/PaperDetailPanel/PaperDetailPanel";
import CitationGraphModal from "../../graph/CitationGraphModal/CitationGraphModal";
import { fetchSessions } from "../../../api/papersApi";
import { useSearch } from "../../../hooks/useSearch";
import { useAppState } from "../../../context/AppStateContext";
import styles from "./MainInterface.module.css";

export default function MainInterface() {
  const { query, papers, loading, error, hasSearched, runSearch } = useSearch();
  const { activePaper, openPaperDetail, closePaperDetail, graphOpen, graphQuery, openGraph, closeGraph } =
    useAppState();

  const [sessions, setSessions] = useState([]);

  const loadSessions = () => {
    fetchSessions()
      .then((data) => setSessions(data.sessions || []))
      .catch(() => setSessions([]));
  };

  useEffect(() => {
    loadSessions();
  }, []);

  const handleSearch = (q) => {
    runSearch(q);
    // refresh session list shortly after so the new search shows up in the sidebar
    setTimeout(loadSessions, 600);
  };

  return (
    <AppShell hasSidebar={sessions.length > 0}>
      <Sidebar sessions={sessions} onSelectSession={handleSearch} />

      <div className={`${styles.centerCol} ${hasSearched ? styles.centerColTop : ""}`}>
        <ChatBar onSearch={handleSearch} loading={loading} initialValue={query} />

        {hasSearched && (
          <ResultsList
            papers={papers}
            query={query}
            loading={loading}
            error={error}
            onSelectPaper={openPaperDetail}
            onOpenGraph={openGraph}
          />
        )}
      </div>

      <PaperDetailPanel paper={activePaper} onClose={closePaperDetail} />
      <CitationGraphModal open={graphOpen} query={graphQuery} onClose={closeGraph} />
    </AppShell>
  );
}
