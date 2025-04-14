function togglePanel(id) {
    const panel = document.getElementById(id);
    panel.classList.toggle("open");
  }
  
  async function send() {
    const input = document.getElementById("chat-input").value;

    // 🌟 STEP 1: Full agent pipeline call
    const response = await fetch("http://localhost:5050/generate-all", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input })
    });
  
    const data = await response.json();
    console.log("Agent response:", data);
  
    // ✅ Display project description (from Agent2)
    document.getElementById("output").innerText = data.full_spec.description;
  
    // 🌟 STEP 2: Request SVG based on full_spec
    const svgRes = await fetch("http://localhost:5050/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data.full_spec)  // Send Agent2 output to SVG generator
    });
  
    const svgBlob = await svgRes.blob();
    const svgURL = URL.createObjectURL(svgBlob);
    document.getElementById("svg-output").innerHTML =
      `<object type="image/svg+xml" data="${svgURL}" width="100%" height="500px"></object>`;
  
    // 🌟 STEP 3: Render code blocks (from Agent3)
    const codeBlocks = data.code_blocks.code_blocks || [];
    const codeDiv = document.getElementById("code-output");
    codeDiv.innerHTML = "";
  
    codeBlocks.forEach(block => {
      const pre = document.createElement("pre");
      pre.innerText = `// ${block.component}\n` + block.code;
      codeDiv.appendChild(pre);
    });
  
  }
  