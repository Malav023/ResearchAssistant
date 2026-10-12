import { apiGet, apiPost } from "./client";

export const fetchTopPapers = (query) =>
  apiGet(`/api/papers?q=${encodeURIComponent(query)}`);

export const fetchSessions = () => apiGet("/api/sessions");

export const saveSession = (query, paperIds) =>
  apiPost("/api/sessions", { query, paper_ids: paperIds });
