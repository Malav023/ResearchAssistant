import { apiGet } from "./client";

export const fetchCitationGraph = (query) =>
  apiGet(`/api/graph?q=${encodeURIComponent(query)}`);
