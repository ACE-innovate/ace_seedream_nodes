import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Applies ACE Seedream Layerize placement to Compositor4 automatically.
// Backend broadcasts "ace_seedream_fabric" with the fabricData JSON; we write it
// into every Compositor4 node's fabricData widget and call the editor's public
// restoreState() (same path Compositor uses on workflow load). Transforms are
// stored as pending and applied as the images arrive.
app.registerExtension({
  name: "ace.seedream.compositor_fabric",
  setup() {
    api.addEventListener("ace_seedream_fabric", (event) => {
      const fabric = event?.detail?.fabric;
      if (!fabric) return;
      const nodes = (app.graph?._nodes || []).filter((n) => n.type === "Compositor4");
      if (!nodes.length) return;
      for (const node of nodes) {
        try {
          const w = node.widgets?.find((x) => x.name === "fabricData");
          if (w) w.value = fabric;
          if (node.editor?.restoreState) node.editor.restoreState(fabric);
          node.setDirtyCanvas?.(true, true);
        } catch (e) {
          console.warn("[ace_seedream] fabric apply failed on node", node.id, e);
        }
      }
      console.log(`[ace_seedream] fabric applied to ${nodes.length} Compositor4 node(s)`);
    });
  },
});
