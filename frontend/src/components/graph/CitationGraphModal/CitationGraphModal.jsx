import { useEffect, useState } from "react";
import Modal from "../../common/Modal/Modal";
import CitationGraphCanvas from "../CitationGraphCanvas/CitationGraphCanvas";
import { fetchCitationGraph } from "../../../api/graphApi";
import styles from "./CitationGraphModal.module.css";

export default function CitationGraphModal({ open, query, onClose }) {
  const [graph, setGraph] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);

  useEffect(() => {
    if (!open || !query) return;

    let cancelled = false;
    setLoading(true);
    setError(null);
    setGraph(null);
    setSelectedNode(null);

    fetchCitationGraph(query)
      .then((data) => {
        if (!cancelled) setGraph(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message || "Failed to load citation graph");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [open, query]);

  return (
    <Modal open={open} onClose={onClose} size="lg" ariaLabel="Citation graph">
      <div className={styles.header}>
        <div>
          <span className={styles.eyebrow}>Citation graph</span>
          <h2 className={styles.title}>“{query}”</h2>
        </div>
        <button className={styles.closeBtn} onClick={onClose} aria-label="Close citation graph" type="button">
          ✕
        </button>
      </div>

      <div className={styles.body}>
        {loading && (
          <div className={styles.state}>
            <span className={styles.spinner} />
            <span>Building citation graph…</span>
          </div>
        )}

        {error && <div className={`${styles.state} ${styles.error}`}>{error}</div>}

        {!loading && !error && graph && graph.nodes.length === 0 && (
          <div className={styles.state}>No graph data found for this query.</div>
        )}

        {!loading && !error && graph && graph.nodes.length > 0 && (
          <div className={styles.canvasArea}>
            <CitationGraphCanvas graph={graph} onSelectNode={setSelectedNode} />

            {selectedNode && (
              <div className={styles.detail}>
                <div className={styles.detailTitle}>{selectedNode.label}</div>
                <div className={styles.detailMeta}>
                  {selectedNode.first_author && <span>{selectedNode.first_author}</span>}
                  {selectedNode.year && <span>{selectedNode.year}</span>}
                  <span>{(selectedNode.citations || 0).toLocaleString()} citations</span>
                </div>
              </div>
            )}

            <div className={styles.legend}>
              <span className={styles.legendItem}>
                <span className={styles.legendLineCitation} /> Citation
              </span>
              <span className={styles.legendItem}>
                <span className={styles.legendLineRelated} /> Related
              </span>
            </div>
          </div>
        )}
      </div>
    </Modal>
  );
}
