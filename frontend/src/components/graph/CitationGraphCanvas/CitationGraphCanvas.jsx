import { useEffect, useRef } from "react";
import * as d3 from "d3";
import styles from "./CitationGraphCanvas.module.css";

const COLORS = {
  citation: "#0070F3",
  related: "#8892B0",
  nodeStroke: "#0070F3",
  nodeFillStart: "#DCE7FB",
  text: "#0A192F",
};

function nodeRadius(d) {
  const c = d.citations || 0;
  if (c > 5000) return 16;
  if (c > 1000) return 13;
  if (c > 200) return 10;
  if (c > 50) return 8;
  return 6;
}

function truncate(str, max) {
  if (!str) return "";
  return str.length > max ? str.slice(0, max - 1) + "…" : str;
}

export default function CitationGraphCanvas({ graph, onSelectNode }) {
  const svgRef = useRef(null);
  const containerRef = useRef(null);
  const tooltipRef = useRef(null);

  useEffect(() => {
    if (!graph || !graph.nodes || graph.nodes.length === 0) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const yearScale = d3
      .scaleSequential()
      .domain([1990, 2025])
      .interpolator(d3.interpolate(COLORS.nodeFillStart, COLORS.nodeStroke));

    const g = svg.append("g");

    const zoom = d3
      .zoom()
      .scaleExtent([0.3, 4])
      .on("zoom", (event) => g.attr("transform", event.transform));
    svg.call(zoom);

    const defs = svg.append("defs");
    ["citation", "related"].forEach((type) => {
      defs
        .append("marker")
        .attr("id", `arrow-${type}`)
        .attr("viewBox", "0 -4 8 8")
        .attr("refX", 18)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-4L8,0L0,4")
        .attr("fill", COLORS[type])
        .attr("opacity", 0.6);
    });

    const link = g
      .append("g")
      .selectAll("line")
      .data(graph.edges)
      .join("line")
      .attr("class", styles.link)
      .attr("stroke", (d) => COLORS[d.type] || COLORS.related)
      .attr("stroke-width", (d) => (d.type === "citation" ? 1.6 : 1))
      .attr("stroke-dasharray", (d) => (d.type === "related" ? "3 3" : null))
      .attr("stroke-opacity", 0.4)
      .attr("marker-end", (d) => `url(#arrow-${d.type})`);

    const nodeGroup = g
      .append("g")
      .selectAll("g")
      .data(graph.nodes)
      .join("g")
      .attr("class", styles.node)
      .call(
        d3
          .drag()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    nodeGroup
      .append("circle")
      .attr("r", nodeRadius)
      .attr("fill", (d) => (d.year ? yearScale(Math.min(Math.max(d.year, 1990), 2025)) : COLORS.nodeFillStart))
      .attr("stroke", COLORS.nodeStroke)
      .attr("stroke-width", 1.5);

    nodeGroup
      .append("text")
      .attr("dy", (d) => nodeRadius(d) + 12)
      .attr("text-anchor", "middle")
      .attr("font-size", "9px")
      .attr("fill", COLORS.text)
      .style("pointer-events", "none")
      .text((d) => truncate(d.label, 22));

    const tooltip = d3.select(tooltipRef.current);

    function focusNode(hoveredId) {
      if (!hoveredId) {
        nodeGroup.classed(styles.dimmed, false);
        link.classed(styles.dimmed, false);
        return;
      }
      const connected = new Set([hoveredId]);
      graph.edges.forEach((e) => {
        const s = typeof e.source === "object" ? e.source.id : e.source;
        const t = typeof e.target === "object" ? e.target.id : e.target;
        if (s === hoveredId) connected.add(t);
        if (t === hoveredId) connected.add(s);
      });
      nodeGroup.classed(styles.dimmed, (d) => !connected.has(d.id));
      link.classed(styles.dimmed, (d) => {
        const s = typeof d.source === "object" ? d.source.id : d.source;
        const t = typeof d.target === "object" ? d.target.id : d.target;
        return s !== hoveredId && t !== hoveredId;
      });
    }

    nodeGroup
      .on("mouseenter", (event, d) => {
        tooltip
          .style("opacity", 1)
          .html(
            `<div class="${styles.ttTitle}">${truncate(d.label || "Untitled", 70)}</div>
             <div class="${styles.ttRow}"><span>Year</span><span>${d.year || "—"}</span></div>
             <div class="${styles.ttRow}"><span>Cited by</span><span>${(d.citations || 0).toLocaleString()}</span></div>`
          );
        focusNode(d.id);
      })
      .on("mousemove", (event) => {
        const [x, y] = d3.pointer(event, container);
        tooltip.style("left", `${x + 16}px`).style("top", `${y - 10}px`);
      })
      .on("mouseleave", () => {
        tooltip.style("opacity", 0);
        focusNode(null);
      })
      .on("click", (event, d) => {
        event.stopPropagation();
        onSelectNode?.(d);
      });

    const simulation = d3
      .forceSimulation(graph.nodes)
      .force(
        "link",
        d3
          .forceLink(graph.edges)
          .id((d) => d.id)
          .distance(90)
          .strength(0.4)
      )
      .force("charge", d3.forceManyBody().strength(-220))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force(
        "collide",
        d3.forceCollide().radius((d) => nodeRadius(d) + 14)
      )
      .on("tick", () => {
        link
          .attr("x1", (d) => d.source.x)
          .attr("y1", (d) => d.source.y)
          .attr("x2", (d) => d.target.x)
          .attr("y2", (d) => d.target.y);
        nodeGroup.attr("transform", (d) => `translate(${d.x},${d.y})`);
      });

    return () => simulation.stop();
  }, [graph, onSelectNode]);

  return (
    <div ref={containerRef} className={styles.canvasWrap}>
      <svg ref={svgRef} className={styles.svg} />
      <div ref={tooltipRef} className={styles.tooltip} />
    </div>
  );
}
