import { useAppState } from "../context/AppStateContext";

/**
 * Central place components read from to know how the main layout should
 * reflow: is the sidebar open, is a paper detail panel open, is the graph
 * modal open. AppShell uses this to compute margins/centering; individual
 * components use it to know whether to render blurred/dimmed.
 */
export function usePanelState() {
  const { sidebarOpen, activePaper, graphOpen } = useAppState();

  const anyOverlayOpen = Boolean(activePaper) || graphOpen;

  return { sidebarOpen, activePaper, graphOpen, anyOverlayOpen };
}
