const cases = [
  {
    id: "20250731_p1",
    trial: "20250731_p1_side_right_1c",
    videoHz: 5.45,
    leftHz: 4.10,
    goldHz: 5.43,
    errorHz: 0.02,
    risk: "中风险",
    riskTitle: "右手检测到稳定节律性震颤特征",
    image: "./assets/spectrum_good.png",
    method: "tips_relative_wrist_pca_validated",
    note: "右手主频稳定，左手为轻度疑似信号。建议结合临床症状复核。"
  },
  {
    id: "20250813_p2",
    trial: "20250813_p2_side_right_1c",
    videoHz: 5.46,
    leftHz: 5.20,
    goldHz: 5.42,
    errorHz: 0.04,
    risk: "高风险",
    riskTitle: "双手均出现稳定震颤频段",
    image: "./assets/spectrum_stable.png",
    method: "tips_relative_wrist_pca_validated",
    note: "双手频率均落在常见帕金森震颤筛查范围，建议神经内科进一步评估。"
  },
  {
    id: "20251022_p2",
    trial: "20251022_p2_side_right_1c",
    videoHz: 3.80,
    leftHz: 0,
    goldHz: 3.87,
    errorHz: 0.07,
    risk: "中风险",
    riskTitle: "单侧检测到低频节律性震颤",
    image: "./assets/spectrum_low.png",
    method: "tips_relative_wrist_pca_validated",
    note: "右手低频震颤较稳定，左手未形成可靠峰值。建议重新采集双手视频确认。"
  },
  {
    id: "20260107_p1",
    trial: "20260107_p1_side_right_1c",
    videoHz: 3.76,
    leftHz: 3.55,
    goldHz: 3.79,
    errorHz: 0.02,
    risk: "中风险",
    riskTitle: "双手存在接近频率的疑似震颤",
    image: "./assets/spectrum_20260107_p1.png",
    method: "tips_relative_wrist_pca_validated",
    note: "双手频率接近，整体稳定性中等。适合进入复测或医生评估流程。"
  }
];

const videoInput = document.getElementById("videoInput");
const previewVideo = document.getElementById("previewVideo");
const emptyPreview = document.getElementById("emptyPreview");
const fileStatus = document.getElementById("fileStatus");
const analyzeBtn = document.getElementById("analyzeBtn");
const progressBar = document.getElementById("progressBar");
const processNote = document.getElementById("processNote");
const steps = [...document.querySelectorAll("#steps li")];
const caseSwitcher = document.getElementById("caseSwitcher");
const rightHz = document.getElementById("rightHz");
const leftHz = document.getElementById("leftHz");
const riskLevel = document.getElementById("riskLevel");
const riskTitle = document.getElementById("riskTitle");
const riskText = document.getElementById("riskText");
const resultText = document.getElementById("resultText");
const spectrumImage = document.getElementById("spectrumImage");
const caseId = document.getElementById("caseId");
const compareBody = document.getElementById("compareBody");

let selectedCase = cases[0];

function renderCase(current) {
  selectedCase = current;
  rightHz.textContent = `${current.videoHz.toFixed(2)} Hz`;
  leftHz.textContent = current.leftHz > 0 ? `${current.leftHz.toFixed(2)} Hz` : "未检出";
  riskLevel.textContent = current.risk;
  riskTitle.textContent = current.riskTitle;
  riskText.textContent = `${current.note} 本页面为静态演示，筛查结论不等同医学诊断。`;
  resultText.textContent = current.note;
  spectrumImage.src = current.image;
  caseId.textContent = current.id;

  [...caseSwitcher.children].forEach((button) => {
    button.classList.toggle("active", button.dataset.caseId === current.id);
  });
}

function renderCases() {
  cases.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.caseId = item.id;
    button.textContent = item.id;
    button.addEventListener("click", () => renderCase(item));
    caseSwitcher.appendChild(button);
  });
  renderCase(cases[0]);
}

function renderTable() {
  compareBody.innerHTML = cases
    .map(
      (item) => `
        <tr>
          <td>${item.trial}</td>
          <td>${item.videoHz.toFixed(2)} Hz</td>
          <td>${item.goldHz.toFixed(2)} Hz</td>
          <td>${item.errorHz.toFixed(2)} Hz</td>
          <td><span class="badge">Accepted</span></td>
          <td>${item.method}</td>
        </tr>
      `
    )
    .join("");
}

videoInput.addEventListener("change", () => {
  const file = videoInput.files?.[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  previewVideo.src = url;
  previewVideo.style.display = "block";
  emptyPreview.style.display = "none";
  fileStatus.textContent = file.name;
  processNote.textContent = "视频已载入，可以开始静态筛查演示。";
});

analyzeBtn.addEventListener("click", () => {
  let step = 0;
  progressBar.style.width = "0%";
  processNote.textContent = "正在模拟双手视频筛查流程...";
  steps.forEach((item) => item.classList.remove("active"));
  steps[0].classList.add("active");

  const timer = window.setInterval(() => {
    step += 1;
    const percent = Math.min(100, Math.round((step / steps.length) * 100));
    progressBar.style.width = `${percent}%`;
    steps.forEach((item, index) => item.classList.toggle("active", index === Math.min(step, steps.length - 1)));

    if (step >= steps.length) {
      window.clearInterval(timer);
      processNote.textContent = `筛查完成：展示 ${selectedCase.id} 的双手视频结果；金标准对比仅在算法验证区展示。`;
      document.getElementById("result").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, 420);
});

renderCases();
renderTable();
