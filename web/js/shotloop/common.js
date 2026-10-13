// Shot Planner: constants and small helpers shared by the panel and its views (no side effects: ComfyUI loads every
// .js under web/js as an extension, so this module only exports).
import { h as vueH, reactive, ref } from "../vendor/vue.esm-browser.prod.mjs";

// UI language is a browser preference, not part of the workflow or generation prompts.
const LANG_KEY = "bfs.shotloop.language";
const savedLanguage = (() => { try { return localStorage.getItem(LANG_KEY) || "en"; } catch { return "en"; } })();
export const uiLanguage = ref(["en", "pt", "es", "zh"].includes(savedLanguage) ? savedLanguage : "en");
export function setUiLanguage(value) {
  if (!["en", "pt", "es", "zh"].includes(value)) return;
  uiLanguage.value = value;
  try { localStorage.setItem(LANG_KEY, value); } catch { /* private mode */ }
}

const TEXT = {
  "🎞 Video": ["🎞 Vídeo", "🎞 Vídeo", "🎞 视频"], "🎬 Shots": ["🎬 Shots", "🎬 Planos", "🎬 镜头"],
  "👥 People & masks": ["👥 Pessoas e máscaras", "👥 Personas y máscaras", "👥 人物和蒙版"],
  "📝 Prompts & refs": ["📝 Prompts e referências", "📝 Prompts y referencias", "📝 提示词和参考图"],
  "▶ Run": ["▶ Executar", "▶ Ejecutar", "▶ 运行"], "📖 Guide": ["📖 Guia", "📖 Guía", "📖 指南"],
  "🧭 Tour": ["🧭 Tour", "🧭 Recorrido", "🧭 引导"], "Language": ["Idioma", "Idioma", "语言"],
  "Source video": ["Vídeo de origem", "Vídeo de origen", "源视频"], "Range": ["Trecho", "Rango", "范围"],
  "Split into shots": ["Dividir em planos", "Dividir en planos", "拆分镜头"], "Generation size": ["Tamanho da geração", "Tamaño de generación", "生成尺寸"],
  "Camera cuts": ["Cortes de câmera", "Cortes de cámara", "镜头切换"], "Fixed length": ["Duração fixa", "Duración fija", "固定时长"],
  "One shot (whole range)": ["Um plano (trecho inteiro)", "Un plano (rango completo)", "单个镜头（整个范围）"],
  "Mode": ["Modo", "Modo", "模式"], "Detector": ["Detector", "Detector", "检测器"], "Sensitivity": ["Sensibilidade", "Sensibilidad", "灵敏度"],
  "Frame grid": ["Grade de frames", "Múltiplo de fotogramas", "帧步长"], "Max seconds / shot": ["Máx. segundos por plano", "Máx. segundos por plano", "每个镜头最长秒数"],
  "Min seconds / shot": ["Mín. segundos por plano", "Mín. segundos por plano", "每个镜头最短秒数"], "Timeline fps": ["FPS da linha do tempo", "FPS de la línea de tiempo", "时间轴帧率"],
  "Megapixels": ["Megapixels", "Megapíxeles", "百万像素"], "Size multiple": ["Múltiplo do tamanho", "Múltiplo del tamaño", "尺寸倍数"],
  "✂ Auto split": ["✂ Dividir automaticamente", "✂ División automática", "✂ 自动拆分"], "↻ Re-analyse": ["↻ Analisar novamente", "↻ Volver a analizar", "↻ 重新分析"],
  "Reference & prompt": ["Referência e prompt", "Referencia y prompt", "参考图和提示词"], "Who is replaced": ["Quem será substituído", "Quién se reemplaza", "替换对象"],
  "Continuity": ["Continuidade", "Continuidad", "连续性"], "Copy to other shots": ["Copiar para outros planos", "Copiar a otros planos", "复制到其他镜头"],
  "Global reference & prompt": ["Referência e prompt globais", "Referencia y prompt globales", "全局参考图和提示词"],
  "Describe references ({details})": ["Descrever referências ({details})", "Describir referencias ({details})", "描述参考图（{details}）"],
  "VLM": ["VLM", "VLM", "视觉语言模型"], "Use the VLM when the workflow runs": ["Usar o VLM ao executar o workflow", "Usar el VLM al ejecutar el flujo", "运行工作流时使用 VLM"],
  "Fill the mask text": ["Preencher o texto da máscara", "Completar el texto de la máscara", "填写蒙版描述"],
  "Fill {shot} in the prompt": ["Preencher {shot} no prompt", "Completar {shot} en el prompt", "在提示词中填入 {shot}"],
  "Frames per shot": ["Frames por plano", "Fotogramas por plano", "每个镜头的帧数"], "Max tokens": ["Máx. tokens", "Máx. tokens", "最大 token 数"],
  "🤖 Analyse shots": ["🤖 Analisar planos", "🤖 Analizar planos", "🤖 分析镜头"], "📝 Describe refs": ["📝 Descrever referências", "📝 Describir referencias", "📝 描述参考图"],
  "🎨 Color Guide": ["🎨 Guia de cores", "🎨 Guía de color", "🎨 颜色引导"], "Ref 1 color": ["Cor da ref. 1", "Color de ref. 1", "参考图 1 的颜色"],
  "Ref 2": ["Ref. 2", "Ref. 2", "参考图 2"], "+ Add a second person (optional)": ["+ Adicionar segunda pessoa (opcional)", "+ Añadir segunda persona (opcional)", "+ 添加第二个人（可选）"],
  "Second person · Ref 2": ["Segunda pessoa · Ref. 2", "Segunda persona · Ref. 2", "第二个人 · 参考图 2"],
  "Describe the second person for SAM 3": ["Descreva a segunda pessoa para o SAM 3", "Describe la segunda persona para SAM 3", "为 SAM 3 描述第二个人"],
  "👁 Preview": ["👁 Prévia", "👁 Vista previa", "👁 预览"], "More options": ["Mais opções", "Más opciones", "更多选项"],
  "Keep ref 2 color consistent across shots (optional)": ["Manter a cor da ref. 2 nos outros planos (opcional)", "Mantener el color de ref. 2 en otros planos (opcional)", "在其他镜头中保持参考图 2 的颜色（可选）"],
  "Ref 1 identity": ["Identidade da ref. 1", "Identidad de ref. 1", "参考图 1 的身份"], "Ref 2 identity": ["Identidade da ref. 2", "Identidad de ref. 2", "参考图 2 的身份"],
  "Choose manually": ["Escolher manualmente", "Elegir manualmente", "手动选择"], "↩ cast color": ["↩ cor do elenco", "↩ color del reparto", "↩ 使用角色颜色"],
  "Opacity": ["Opacidade", "Opacidad", "不透明度"], "The default keeps movement and expression visible.": ["O padrão mantém o movimento e a expressão visíveis.", "El valor predeterminado mantiene visibles el movimiento y la expresión.", "默认值会保留动作和表情。"],
  "Full frame": ["Quadro inteiro", "Fotograma completo", "整帧"], "🧩 Frame + paste": ["🧩 Quadro + colar", "🧩 Fotograma + pegar", "🧩 整帧生成并贴回"],
  "🎭 Mask only": ["🎭 Somente máscara", "🎭 Solo máscara", "🎭 仅蒙版"], "✂ Crop": ["✂ Recorte", "✂ Recorte", "✂ 裁剪"], "✂🎭 Crop + mask": ["✂🎭 Recorte + máscara", "✂🎭 Recorte + máscara", "✂🎭 裁剪并蒙版"],
  "People & masks": ["Pessoas e máscaras", "Personas y máscaras", "人物和蒙版"], "Cast": ["Elenco", "Reparto", "角色"],
  "Mask & crop settings": ["Configurações de máscara e recorte", "Ajustes de máscara y recorte", "蒙版和裁剪设置"],
  "Color Guide identity color": ["Cor de identidade do Guia de cores", "Color de identidad de la Guía de color", "颜色引导身份颜色"],
  "No Color Guide color": ["Sem cor", "Sin color", "无颜色引导颜色"], "Green": ["Verde", "Verde", "绿色"], "Purple": ["Roxo", "Morado", "紫色"],
  "Red": ["Vermelho", "Rojo", "红色"], "Blue": ["Azul", "Azul", "蓝色"], "Yellow": ["Amarelo", "Amarillo", "黄色"], "Magenta": ["Magenta", "Magenta", "品红"], "Cyan": ["Ciano", "Cian", "青色"],
  "Checks": ["Verificações", "Comprobaciones", "检查"], "Run": ["Executar", "Ejecutar", "运行"], "Filters": ["Filtros", "Filtros", "筛选"],
  "Auto loop (one run)": ["Loop automático (uma execução)", "Bucle automático (una ejecución)", "自动循环（一次运行）"],
  "Queue loop (one shot per run)": ["Fila (um plano por execução)", "Cola (un plano por ejecución)", "队列循环（每次运行一个镜头）"],
  "Skipped shots in the output": ["Planos ignorados na saída", "Planos omitidos en la salida", "输出中跳过的镜头"], "Keep original video": ["Manter vídeo original", "Mantener vídeo original", "保留原视频"],
  "Remove them": ["Removê-los", "Quitarlos", "移除"], "Max shots (0 = all)": ["Máx. planos (0 = todos)", "Máx. planos (0 = todos)", "镜头上限（0 = 全部）"],
  "Max total seconds (0 = all)": ["Máx. duração total (0 = tudo)", "Duración total máx. (0 = todo)", "总时长上限（0 = 全部）"],
  "Auto-queue the next shot": ["Enfileirar próximo plano automaticamente", "Añadir automáticamente el siguiente plano a la cola", "自动将下一个镜头加入队列"],
  "Needs a person": ["Precisa ter uma pessoa", "Debe haber una persona", "需要有人物"], "Needs a face": ["Precisa ter um rosto", "Debe haber un rostro", "需要有人脸"],
  "Skip dark / fades": ["Ignorar cenas escuras / fades", "Omitir escenas oscuras / fundidos", "跳过暗场/淡入淡出"], "Skip static shots": ["Ignorar planos estáticos", "Omitir planos estáticos", "跳过静止镜头"],
  "👤 Analyse people & faces": ["👤 Analisar pessoas e rostos", "👤 Analizar personas y rostros", "👤 分析人物和人脸"],
  "Video": ["Vídeo", "Vídeo", "视频"], "Shots": ["Planos", "Planos", "镜头"], "People": ["Pessoas", "Personas", "人物"], "Prompts": ["Prompts", "Prompts", "提示词"],
  "Back": ["Voltar", "Atrás", "上一步"], "Next": ["Avançar", "Siguiente", "下一步"], "Close tour": ["Fechar tour", "Cerrar recorrido", "关闭引导"],
  "Shot Planner tour": ["Tour do Shot Planner", "Recorrido de Shot Planner", "Shot Planner 引导"], "Shot": ["Plano", "Plano", "镜头"],
  "This tour is optional. Use Next to visit each main tab; close it whenever you like.": ["Este tour é opcional. Use Avançar para conhecer cada aba; feche quando quiser.", "Este recorrido es opcional. Pulsa Siguiente para visitar cada pestaña; puedes cerrarlo cuando quieras.", "此引导为可选。点击下一步浏览各个主选项卡，也可随时关闭。"],
  "Choose a video, set the frame range, split at camera cuts, then set the generation size.": ["Escolha um vídeo e o trecho, divida nos cortes de câmera e defina o tamanho da geração.", "Elige un vídeo y un rango, divídelo en cortes de cámara y define el tamaño de generación.", "选择视频和范围，按镜头切换拆分，然后设置生成尺寸。"],
  "Review each shot, assign its references and prompt, and choose how the result is applied.": ["Revise cada plano, associe referências e prompt e escolha como aplicar o resultado.", "Revisa cada plano, asigna referencias y prompt, y elige cómo aplicar el resultado.", "检查每个镜头，设置参考图和提示词，并选择结果的应用方式。"],
  "Link detected people to references. Set masks here or per shot; ArcFace links are a guide and need your review.": ["Associe as pessoas detectadas às referências. Configure máscaras aqui ou por plano; revise os vínculos ArcFace.", "Vincula las personas detectadas con referencias. Configura máscaras aquí o por plano; revisa los enlaces ArcFace.", "将检测到的人物关联到参考图。可在此处或逐镜头设置蒙版；请检查 ArcFace 关联。"],
  "Set shared references, prompts and optional VLM assistance. Written prompts are kept exactly as authored.": ["Defina referências, prompts e assistência opcional do VLM. Os prompts escritos são preservados como estão.", "Configura referencias, prompts y asistencia opcional del VLM. Los prompts escritos se conservan tal cual.", "设置共享参考图、提示词和可选的 VLM 辅助。提示词会保持原样。"],
  "Check warnings, choose auto or queue loop, and set filters or quick-run limits.": ["Confira avisos, escolha loop automático ou fila e configure filtros e limites de teste.", "Revisa avisos, elige bucle automático o cola y configura filtros y límites de prueba.", "查看警告，选择自动循环或队列，并设置筛选条件和快速运行限制。"],
  "Color Guide adds a translucent marker to a selected person. Ref 1 and Ref 2 must use different colors in the same shot.": ["O Guia de cores aplica uma cor translúcida à pessoa selecionada. Ref. 1 e Ref. 2 precisam de cores diferentes no mesmo plano.", "La Guía de color aplica una marca translúcida a la persona seleccionada. Ref. 1 y Ref. 2 deben tener colores distintos en el mismo plano.", "颜色引导会给选中的人物添加半透明颜色。同一镜头中的参考图 1 和参考图 2 必须使用不同颜色。"],
  "🎬 Shot Planner": ["🎬 Planejador de planos", "🎬 Planificador de planos", "🎬 镜头规划器"],
  "— choose a video from the input folder —": ["— escolha um vídeo da pasta de entrada —", "— elige un vídeo de la carpeta de entrada —", "— 从输入文件夹选择视频 —"],
  "⬆ Upload": ["⬆ Enviar", "⬆ Subir", "⬆ 上传"], "Pick or upload a video to start": ["Escolha ou envie um vídeo para começar", "Elige o sube un vídeo para empezar", "选择或上传视频以开始"],
  "It is split into shots the model can generate; every shot then gets its own reference, prompt and mask.": ["O vídeo será dividido em planos que o modelo consegue gerar; cada plano terá sua própria referência, prompt e máscara.", "El vídeo se divide en planos que el modelo puede generar; cada uno tendrá su referencia, prompt y máscara.", "视频会拆分为模型可生成的镜头，每个镜头都有自己的参考图、提示词和蒙版。"],
  "work on a part of the video": ["trabalhe em um trecho do vídeo", "trabaja en una parte del vídeo", "处理视频的一部分"],
  "Start frame": ["Frame inicial", "Fotograma inicial", "起始帧"], "End frame (0 = end)": ["Frame final (0 = fim)", "Fotograma final (0 = final)", "结束帧（0 = 末尾）"],
  "⇤ start here": ["⇤ começar aqui", "⇤ empezar aquí", "⇤ 从此处开始"], "⇥ end here": ["⇥ terminar aqui", "⇥ terminar aquí", "⇥ 在此处结束"], "✕ whole video": ["✕ vídeo inteiro", "✕ vídeo completo", "✕ 整个视频"],
  "Camera cuts or fixed length, on the model's frame grid": ["Cortes de câmera ou duração fixa, na grade de frames do modelo", "Cortes de cámara o duración fija, según la cuadrícula de fotogramas del modelo", "按镜头切换或固定时长，并符合模型帧步长"],
  "re-analyses": ["reanalisar", "vuelve a analizar", "重新分析"], "ignores Max seconds / shot": ["ignora Máx. segundos por plano", "ignora Máx. segundos por plano", "忽略每个镜头的最长秒数"],
  "cropped shots use it for the crop": ["planos recortados usam esse valor no recorte", "los planos recortados usan este valor", "裁剪镜头会使用此尺寸"],
  "Shots use their main person's reference": ["Usar a referência da pessoa principal do plano", "Usar la referencia de la persona principal del plano", "镜头使用主要人物的参考图"],
  "Split where the main person changes": ["Dividir quando a pessoa principal mudar", "Dividir cuando cambie la persona principal", "主要人物变化时拆分"],
  "Only run shots with a linked person": ["Executar apenas planos com pessoa vinculada", "Ejecutar solo planos con una persona vinculada", "仅运行已关联人物的镜头"],
  "a shot without its own reference takes the reference of the person with the most screen time in it": ["um plano sem referência própria usa a referência da pessoa que aparece por mais tempo nele", "un plano sin referencia propia usa la de la persona que aparece más tiempo", "没有单独参考图的镜头会使用出镜时间最长人物的参考图"],
  "adds a boundary when the biggest face switches to someone else (re-splits the plan)": ["cria uma divisão quando o maior rosto muda para outra pessoa (refaz a divisão)", "añade un corte cuando el rostro más grande cambia de persona (vuelve a dividir)", "最大人脸切换到其他人时添加分界（重新拆分计划）"],
  "shots where no linked person appears are skipped": ["planos sem pessoa vinculada são ignorados", "se omiten los planos sin una persona vinculada", "跳过没有关联人物的镜头"],
  "Nobody found yet.": ["Nenhuma pessoa encontrada.", "Aún no se encontraron personas.", "尚未检测到人物。"], "↻ Re-analyse people": ["↻ Analisar pessoas novamente", "↻ Volver a analizar personas", "↻ 重新分析人物"], "👥 Find people": ["👥 Encontrar pessoas", "👥 Buscar personas", "👥 查找人物"],
  "faces sampled every 0.5 s and grouped by identity (InsightFace, CPU)": ["rostos amostrados a cada 0,5 s e agrupados por identidade (InsightFace, CPU)", "rostros muestreados cada 0,5 s y agrupados por identidad (InsightFace, CPU)", "每 0.5 秒采样人脸并按身份分组（InsightFace，CPU）"],
  "detects faces across the video and groups them into people": ["detecta rostos no vídeo e os agrupa por pessoa", "detecta rostros en el vídeo y los agrupa por persona", "检测视频中的人脸并按人物分组"],
  "track faces, link each person to a reference, split and assign shots by who is on screen": ["rastreie rostos, associe cada pessoa a uma referência e divida os planos por quem aparece", "rastrea rostros, vincula cada persona a una referencia y divide los planos según quién aparece", "跟踪人脸、关联参考图，并按出镜人物拆分镜头"],
  "Each shot picks what it marks (🎯 text or points, or 🎞 its own mask video) and how it is generated (": ["Cada plano define a seleção (🎯 texto, pontos ou 🎞 vídeo de máscara) e o modo de geração (", "Cada plano define la selección (🎯 texto, puntos o 🎞 vídeo de máscara) y el modo de generación (", "每个镜头设置选区（🎯 文本、点或 🎞 蒙版视频）和生成方式（"],
  "🎞 Mask video (whole video)": ["🎞 Vídeo de máscara (vídeo inteiro)", "🎞 Vídeo de máscara (vídeo completo)", "🎞 蒙版视频（整个视频）"],
  "— none: SAM 3 per shot —": ["— nenhum: SAM 3 por plano —", "— ninguno: SAM 3 por plano —", "— 无：每个镜头使用 SAM 3 —"], "Fill holes": ["Preencher buracos", "Rellenar huecos", "填充孔洞"], "Invert": ["Inverter", "Invertir", "反转"],
  "Show masks on shots": ["Mostrar máscaras nos planos", "Mostrar máscaras en los planos", "在镜头卡片上显示蒙版"], "Expand (px)": ["Expandir (px)", "Expandir (px)", "扩展（像素）"],
  "Temporal expand (frames)": ["Expansão temporal (frames)", "Expansión temporal (fotogramas)", "时间扩展（帧）"], "Detection threshold": ["Limiar de detecção", "Umbral de detección", "检测阈值"], "Max objects": ["Máx. objetos", "Máx. objetos", "最大对象数"],
  "Crop padding (%)": ["Margem do recorte (%)", "Margen del recorte (%)", "裁剪边距（%）"], "Feather (px)": ["Suavização da borda (px)", "Suavizado del borde (px)", "羽化（像素）"], "Paste back": ["Colar de volta", "Pegar de vuelta", "贴回方式"],
  "only the mask (feathered)": ["somente a máscara (suavizada)", "solo la máscara (suavizada)", "仅蒙版（羽化边缘）"], "the whole box (feathered)": ["a caixa inteira (suavizada)", "todo el cuadro (suavizado)", "整个框（羽化边缘）"], "Blockify (px, 0 = off)": ["Blocos (px, 0 = desligado)", "Bloques (px, 0 = desactivado)", "块化（像素，0 = 关闭）"],
  "Uses the official sam3.1_multiplex_fp16 checkpoint (models/checkpoints), downloaded from Comfy-Org/sam3.1 the first time.": ["Usa o checkpoint oficial sam3.1_multiplex_fp16 (models/checkpoints), baixado de Comfy-Org/sam3.1 na primeira utilização.", "Usa el checkpoint oficial sam3.1_multiplex_fp16 (models/checkpoints), descargado de Comfy-Org/sam3.1 la primera vez.", "使用官方 sam3.1_multiplex_fp16 检查点（models/checkpoints），首次运行时从 Comfy-Org/sam3.1 下载。"],
  "SAM 3": ["SAM 3", "SAM 3", "SAM 3"], "track faces, link each person to a reference, split and assign shots by who is on screen": ["rastreie rostos, associe cada pessoa a uma referência e organize os planos por quem aparece", "rastrea rostros, vincula cada persona a una referencia y organiza los planos según quién aparece", "跟踪人脸、关联参考图，并按出镜人物分配镜头"],
  "Global reference & prompt": ["Referência e prompt globais", "Referencia y prompt globales", "全局参考图和提示词"], "defaults for every shot": ["padrão para todos os planos", "valor predeterminado para cada plano", "所有镜头的默认设置"],
  "who the shot becomes": ["em quem o plano será transformado", "en quién se transforma el plano", "镜头中的人物将替换为谁"], "Prompt for this shot (empty = global prompt). Placeholders: {target} {details} {shot} {setting}": ["Prompt deste plano (vazio = prompt global). Variáveis: {target} {details} {shot} {setting}", "Prompt de este plano (vacío = prompt global). Variables: {target} {details} {shot} {setting}", "此镜头的提示词（留空则使用全局提示词）。占位符：{target} {details} {shot} {setting}"],
  "recent (click = ref, shift+click = ref 2):": ["recentes (clique = ref., Shift+clique = ref. 2):", "recientes (clic = ref., Mayús+clic = ref. 2):", "最近使用（点击 = 参考图，Shift+点击 = 参考图 2）："], "✕ clear": ["✕ limpar", "✕ borrar", "✕ 清除"], "✕ clear all": ["✕ limpar tudo", "✕ borrar todo", "✕ 全部清除"], "Use global": ["Usar global", "Usar global", "使用全局设置"],
  "empty slots use the person's reference (Cast) or the global one (Prompts & refs tab)": ["campos vazios usam a referência da pessoa (Elenco) ou a global (aba Prompts e referências)", "los campos vacíos usan la referencia de la persona (Reparto) o la global (Prompts y referencias)", "空白项使用人物参考图（角色）或全局参考图（提示词和参考图选项卡）"],
  "Full frame": ["Quadro inteiro", "Fotograma completo", "整帧"], "The whole frame is regenerated. The mask, if any, only feeds {target} / the setting picture.": ["O quadro inteiro é gerado novamente. A máscara, se houver, só fornece {target} / a imagem de cenário.", "Se vuelve a generar todo el fotograma. La máscara, si existe, solo aporta {target} / la imagen del escenario.", "重新生成整帧。蒙版（如有）仅用于提供 {target} / 场景参考图。"],
  "Only the mask is regenerated, on the whole frame: the rest stays exactly as it was. No crop / uncrop.": ["Somente a área da máscara é regenerada; o restante permanece igual. Sem recorte.", "Solo se regenera el área de la máscara; el resto permanece igual. Sin recorte.", "仅重新生成蒙版区域，其余部分保持不变。不进行裁剪。"],
  "A box around the mask is regenerated (more pixels for a small subject); BFS Shot Join pastes it back.": ["A caixa ao redor da máscara é regenerada (mais pixels para pessoas pequenas); o BFS Shot Join cola o resultado de volta.", "Se regenera el cuadro alrededor de la máscara (más píxeles para sujetos pequeños); BFS Shot Join lo pega de vuelta.", "重新生成蒙版周围的区域（为较小主体提供更多像素）；BFS Shot Join 会将其贴回。"],
  "Inside the crop, only the mask is regenerated: the most detail with the background kept.": ["Dentro do recorte, somente a máscara é regenerada: mais detalhes e fundo preservado.", "Dentro del recorte, solo se regenera la máscara: más detalle y fondo conservado.", "仅重新生成裁剪区域内的蒙版：细节更多，同时保留背景。"],
  "The whole frame is generated (the pose follows the guide, the new subject may be bigger); BFS Shot Join pastes only the subject onto the original: background exact.": ["O quadro inteiro é gerado (a pose segue o guia e o novo sujeito pode ser maior); o BFS Shot Join cola apenas o sujeito no original e preserva o fundo.", "Se genera todo el fotograma (la pose sigue la guía y el sujeto puede ser más grande); BFS Shot Join pega solo al sujeto sobre el original y conserva el fondo.", "生成整帧（姿势遵循引导，主体可更大）；BFS Shot Join 仅将主体贴回原画面，背景保持原样。"],
  "🎯 Points…": ["🎯 Pontos…", "🎯 Puntos…", "🎯 点…"], "👁 Preview": ["👁 Prévia", "👁 Vista previa", "👁 预览"], "✕ Clear": ["✕ Limpar", "✕ Borrar", "✕ 清除"],
  "🎞 mask video": ["🎞 vídeo de máscara", "🎞 vídeo de máscara", "🎞 蒙版视频"], "↳ global prompt": ["↳ prompt global", "↳ prompt global", "↳ 全局提示词"], "— no prompt —": ["— sem prompt —", "— sin prompt —", "— 无提示词 —"],
  "▶ Play all": ["▶ Reproduzir todos", "▶ Reproducir todos", "▶ 播放全部"], "■ Stop": ["■ Parar", "■ Detener", "■ 停止"], "loop shot": ["repetir plano", "repetir plano", "循环播放镜头"],
  "Click the timeline to seek. Play all skips shots that will not run.": ["Clique na linha do tempo para buscar. Reproduzir todos ignora planos que não serão executados.", "Haz clic en la línea de tiempo para buscar. Reproducir todos omite los planos que no se ejecutarán.", "点击时间轴进行定位。播放全部会跳过不运行的镜头。"],
  "Reference & prompt": ["Referência e prompt", "Referencia y prompt", "参考图和提示词"], "Who is replaced": ["Quem será substituído", "Quién se reemplaza", "替换对象"], "Continuity": ["Continuidade", "Continuidad", "连续性"],
  "Global reference & prompt": ["Referência e prompt globais", "Referencia y prompt globales", "全局参考图和提示词"], "Describe references ({details})": ["Descrever referências ({details})", "Describir referencias ({details})", "描述参考图（{details}）"],
  "Custom": ["Personalizado", "Personalizado", "自定义"], "short": ["curto", "breve", "简短"], "full body": ["corpo inteiro", "cuerpo entero", "全身"], "head / face": ["cabeça / rosto", "cabeza / rostro", "头部/面部"], "face attributes": ["características do rosto", "rasgos faciales", "面部特征"], "outfit": ["roupa", "ropa", "服装"],
  "Apply suggestions → all": ["Aplicar sugestões → todos", "Aplicar sugerencias → todos", "应用建议 → 全部"], "fit the prompt": ["ajustar ao prompt", "ajustar al prompt", "适配提示词"],
  "Connect a Qwen3-VL (CLIPLoader) to the planner's vlm input. The VLM buttons work after the workflow has run once with it connected (ComfyUI only hands models to nodes when they run). Each shot's suggestion shows in its editor.": ["Conecte um Qwen3-VL (CLIPLoader) à entrada vlm do planner. Os botões do VLM funcionam depois que o workflow executar uma vez com ele conectado. A sugestão de cada plano aparece no editor.", "Conecta un Qwen3-VL (CLIPLoader) a la entrada vlm del planificador. Los botones del VLM funcionan después de ejecutar el flujo una vez con él conectado. La sugerencia de cada plano aparece en su editor.", "将 Qwen3-VL（CLIPLoader）连接到规划器的 vlm 输入。工作流连接后运行一次，VLM 按钮即可使用。每个镜头的建议会显示在其编辑器中。"],
  "look at every shot and suggests the mask, a description and whether to run it": ["analisa cada plano e sugere máscara, descrição e se deve ser executado", "analiza cada plano y sugiere una máscara, una descripción y si debe ejecutarse", "分析每个镜头并建议蒙版、描述及是否运行"],
  "🎯 Track → whole video": ["🎯 Rastrear → vídeo inteiro", "🎯 Seguir → vídeo completo", "🎯 跟踪 → 整个视频"], "Save (this shot)": ["Salvar (este plano)", "Guardar (este plano)", "保存（此镜头）"], "Save → all shots": ["Salvar → todos os planos", "Guardar → todos los planos", "保存 → 所有镜头"], "Cancel": ["Cancelar", "Cancelar", "取消"], "Clear points": ["Limpar pontos", "Borrar puntos", "清除点"], "👁 Segment": ["👁 Segmentar", "👁 Segmentar", "👁 分割"],
  "Apply these points to EVERY shot, each on its frame at the same relative position (a subject that stays in place)": ["Aplicar estes pontos a TODOS os planos, na mesma posição relativa (para um sujeito que não muda de lugar)", "Aplicar estos puntos a TODOS los planos, en la misma posición relativa (si el sujeto no se mueve)", "将这些点应用到每个镜头的相同相对位置（适用于位置不变的主体）"],
  "the range starts at the player's frame": ["o trecho começa no frame atual do player", "el rango empieza en el fotograma actual", "范围从播放器当前帧开始"], "the range ends after the player's frame": ["o trecho termina após o frame atual do player", "el rango termina después del fotograma actual", "范围在播放器当前帧之后结束"],
  "Nothing to fix in the shots that run.": ["Nenhum problema nos planos que serão executados.", "No hay nada que corregir en los planos que se ejecutarán.", "将运行的镜头没有需要修复的问题。"], "all good": ["tudo certo", "todo bien", "一切正常"], "warning(s)": ["aviso(s)", "aviso(s)", "条警告"], "note(s)": ["observação(ões)", "nota(s)", "条备注"],
  "quick tests": ["testes rápidos", "pruebas rápidas", "快速测试"], "from the start of the range": ["a partir do início do trecho", "desde el inicio del rango", "从范围起点开始"], "Each run generates one shot and stores it. Nodes after BFS Shot Join only run on the last shot, with the full video.": ["Cada execução gera e salva um plano. Os nodes após BFS Shot Join só executam no último plano, com o vídeo completo.", "Cada ejecución genera y guarda un plano. Los nodos posteriores a BFS Shot Join solo se ejecutan en el último plano, con el vídeo completo.", "每次运行生成并保存一个镜头。BFS Shot Join 后的节点仅在最后一个镜头运行，并接收完整视频。"],
  "skipped shots do not run": ["planos ignorados não são executados", "los planos omitidos no se ejecutan", "跳过的镜头不会运行"], "only shots that run": ["somente planos executados", "solo los planos que se ejecutan", "仅统计将运行的镜头"],
  "off": ["desligado", "desactivado", "关闭"], "on": ["ligado", "activado", "开启"], "run": ["executar", "ejecutar", "运行"], "skip": ["ignorar", "omitir", "跳过"], "filters decide": ["filtros decidem", "deciden los filtros", "由筛选条件决定"], "always run": ["sempre executar", "ejecutar siempre", "始终运行"], "always skip": ["sempre ignorar", "omitir siempre", "始终跳过"], "⏸ Disable": ["⏸ Desativar", "⏸ Desactivar", "⏸ 禁用"], "▶ Enable": ["▶ Ativar", "▶ Activar", "▶ 启用"], "✂ Split": ["✂ Dividir", "✂ Dividir", "✂ 拆分"], "⇥ Merge": ["⇥ Unir", "⇥ Unir", "⇥ 合并"],
  "Advanced": ["Avançado", "Avanzado", "高级"], "English": ["Inglês", "Inglés", "英语"], "Português": ["Português", "Portugués", "葡萄牙语"], "Español": ["Espanhol", "Español", "西班牙语"], "中文": ["Chinês", "Chino", "中文"],
  "Color Guide": ["Guia de cores", "Guía de color", "颜色引导"], "Video": ["Vídeo", "Vídeo", "视频"], "Shots": ["Planos", "Planos", "镜头"], "Prompts": ["Prompts", "Prompts", "提示词"],
  "Audio": ["Áudio", "Audio", "音频"], "use the video's audio": ["usar o áudio do vídeo", "usar el audio del vídeo", "使用视频音频"], "silent track": ["faixa silenciosa", "pista silenciosa", "静音音轨"],
  "auto": ["automático", "automático", "自动"], "fixed": ["fixo", "fijo", "固定"], "single": ["único", "único", "单个"], "adaptive": ["adaptativo", "adaptativo", "自适应"], "content": ["conteúdo", "contenido", "内容"], "builtin": ["integrado", "integrado", "内置"],
  "PySceneDetect adaptive": ["PySceneDetect adaptativo", "PySceneDetect adaptativo", "PySceneDetect 自适应"], "PySceneDetect content": ["PySceneDetect por conteúdo", "PySceneDetect por contenido", "PySceneDetect 内容检测"], "Built-in": ["Integrado", "Integrado", "内置"],
  "Auto loop": ["Loop automático", "Bucle automático", "自动循环"], "Queue loop": ["Loop em fila", "Bucle en cola", "队列循环"], "queue loop": ["loop em fila", "bucle en cola", "队列循环"], "auto loop": ["loop automático", "bucle automático", "自动循环"],
  "no audio": ["sem áudio", "sin audio", "无音频"], "silent track": ["faixa silenciosa", "pista silenciosa", "静音音轨"], "timeline": ["linha do tempo", "línea de tiempo", "时间轴"], "generate at": ["gerar em", "generar a", "生成尺寸"],
  "Max person size (0-1 of frame)": ["Tamanho mínimo da pessoa (0–1 do quadro)", "Tamaño mínimo de persona (0–1 del fotograma)", "人物最小尺寸（占画面 0–1）"], "Min person size (0-1 of frame)": ["Tamanho mínimo da pessoa (0–1 do quadro)", "Tamaño mínimo de persona (0–1 del fotograma)", "人物最小尺寸（占画面 0–1）"],
  "Max people (0 = any)": ["Máx. pessoas (0 = qualquer)", "Máx. personas (0 = cualquiera)", "人物上限（0 = 不限）"], "Min frames (0 = off)": ["Mín. frames (0 = desligado)", "Mín. fotogramas (0 = desactivado)", "最少帧数（0 = 关闭）"], "Frames sampled / shot": ["Frames amostrados por plano", "Fotogramas muestreados por plano", "每个镜头采样帧数"], "Dark below (0-1)": ["Escuro abaixo de (0–1)", "Oscuro por debajo de (0–1)", "暗度低于（0–1）"], "Static below": ["Estático abaixo de", "Estático por debajo de", "静止度低于"],
  "e.g. 0.03 skips wide shots": ["ex.: 0,03 ignora planos abertos", "p. ej., 0,03 omite planos amplios", "例如 0.03 会跳过远景镜头"], "skip crowds": ["ignorar multidões", "omitir multitudes", "跳过人群"], "re-analyses": ["reanalisar", "vuelve a analizar", "重新分析"],
  "Describe target": ["Descrever alvo", "Describir objetivo", "描述目标"], "🧑 Describe target": ["🧑 Descrever alvo", "🧑 Describir objetivo", "🧑 描述目标"], "new subject": ["novo sujeito", "nuevo sujeto", "新主体"], "person (or dog, cat, car… what the result shows)": ["pessoa (ou cão, gato, carro… o que aparecerá no resultado)", "persona (o perro, gato, coche… lo que aparecerá en el resultado)", "人物（或狗、猫、汽车等结果中的主体）"],
  "The second person (optional)": ["Segunda pessoa (opcional)", "Segunda persona (opcional)", "第二个人（可选）"], "Follow cast": ["Seguir identidade", "Seguir reparto", "跟随角色"], "manual identity": ["identidade manual", "identidad manual", "手动身份"],
  "Guide video": ["Vídeo guia", "Vídeo guía", "引导视频"], "Reference": ["Referência", "Referencia", "参考图"], "Prompt": ["Prompt", "Prompt", "提示词"], "reference": ["referência", "referencia", "参考图"],
  "on": ["ativado", "activado", "开启"], "off": ["desativado", "desactivado", "关闭"], "not analysed": ["não analisado", "sin analizar", "未分析"], "not analyzed": ["não analisado", "sin analizar", "未分析"],
  "No shots yet": ["Nenhum plano ainda", "Aún no hay planos", "尚无镜头"], "No video": ["Nenhum vídeo", "Sin vídeo", "没有视频"], "No faces": ["Nenhum rosto", "Sin rostros", "没有人脸"], "no faces": ["sem rostos", "sin rostros", "没有人脸"],
  "cut": ["corte", "corte", "切换"], "run": ["executar", "ejecutar", "运行"], "skip": ["ignorar", "omitir", "跳过"], "skip this shot": ["ignorar este plano", "omitir este plano", "跳过此镜头"],
  "people": ["pessoas", "personas", "人物"], "person": ["pessoa", "persona", "人物"], "shots": ["planos", "planos", "镜头"], "shot(s)": ["plano(s)", "plano(s)", "个镜头"], "linked": ["vinculada", "vinculada", "已关联"], "main": ["principal", "principal", "主要"],
  "No VLM yet": ["Nenhum VLM carregado", "Aún no hay un VLM", "尚未加载 VLM"], "Connect a Qwen3-VL": ["Conecte um Qwen3-VL", "Conecta un Qwen3-VL", "连接 Qwen3-VL"],
  "↻ Status": ["↻ Status", "↻ Estado", "↻ 状态"], "⟲ Reset loop": ["⟲ Reiniciar loop", "⟲ Reiniciar bucle", "⟲ 重置循环"], "Queue loop": ["Loop em fila", "Bucle en cola", "队列循环"],
  "Start here": ["Começar aqui", "Empezar aquí", "从此处开始"], "End here": ["Terminar aqui", "Terminar aquí", "在此处结束"], "reset": ["redefinir", "restablecer", "重置"], "all shots": ["todos os planos", "todos los planos", "所有镜头"], "this shot": ["este plano", "este plano", "此镜头"],
  "Describe target": ["Descrever alvo", "Describir objetivo", "描述目标"], "Settings": ["Configurações", "Ajustes", "设置"], "mask text": ["texto da máscara", "texto de máscara", "蒙版描述"],
  "The global reference and prompt are used when a shot has no values of its own.": ["A referência e o prompt globais são usados quando um plano não tem valores próprios.", "La referencia y el prompt globales se usan cuando un plano no tiene valores propios.", "镜头没有单独设置时会使用全局参考图和提示词。"],
  "A VLM is optional": ["O VLM é opcional", "El VLM es opcional", "VLM 为可选项"], "Optional": ["Opcional", "Opcional", "可选"], "optional": ["opcional", "opcional", "可选"],
  "person in white": ["pessoa de branco", "persona de blanco", "穿白衣的人"], "woman in pink top": ["mulher de blusa rosa", "mujer con blusa rosa", "穿粉色上衣的女人"],
  "nothing found": ["nada encontrado", "no se encontró nada", "未找到内容"], "not found": ["não encontrado", "no encontrado", "未找到"], "Preview": ["Prévia", "Vista previa", "预览"],
  "Mask only / Crop + mask": ["Somente máscara / Recorte + máscara", "Solo máscara / Recorte + máscara", "仅蒙版 / 裁剪并蒙版"], "per shot (planner)": ["por plano (planner)", "por plano (planificador)", "逐镜头（规划器）"],
  "Person": ["Pessoa", "Persona", "人物"], "shot": ["plano", "plano", "镜头"], "shot(s)": ["plano(s)", "plano(s)", "个镜头"],
  "mask covers": ["máscara cobre", "la máscara cubre", "蒙版覆盖"], "nothing to fix": ["nada a corrigir", "nada que corregir", "无需修复"],
  "Camera cuts: shots start at the detected cuts, long ones are split evenly. Fixed length: equal parts no longer than the maximum.": ["Cortes de câmera: os planos começam nos cortes detectados; planos longos são divididos igualmente. Duração fixa: partes iguais limitadas pelo máximo.", "Cortes de cámara: los planos empiezan en los cortes detectados; los largos se dividen por igual. Duración fija: partes iguales con el máximo indicado.", "镜头切换：镜头从检测到的切点开始，较长镜头会均匀拆分。固定时长：拆分为不超过上限的等长片段。"],
  "Cut detector. PySceneDetect adaptive works on most footage (fast camera moves don't trigger it); content is stricter on hard cuts; built-in needs no package.": ["Detector de cortes. O modo adaptativo do PySceneDetect funciona na maioria dos vídeos (movimentos rápidos de câmera não acionam cortes); o modo conteúdo é mais rigoroso; o integrado não exige pacote.", "Detector de cortes. El modo adaptativo de PySceneDetect funciona en la mayoría de vídeos (los movimientos rápidos no activan cortes); contenido es más estricto; el integrado no requiere paquetes.", "镜头切换检测器。PySceneDetect 自适应模式适用于大多数素材（快速运镜不会触发切点）；内容模式对硬切更严格；内置模式无需额外软件包。"],
  "How easily a change counts as a cut. Higher finds more cuts (and more false ones).": ["O quanto uma mudança é considerada corte. Um valor maior encontra mais cortes e também mais falsos positivos.", "Facilidad con la que un cambio cuenta como corte. Un valor mayor detecta más cortes y también más falsos positivos.", "变化被判定为切点的难易程度。数值越高，检测到的切点越多，误报也越多。"],
  "Frame counts the model accepts. H3: 17n+5, LTX/Wan: 8n+1, Wan: 4n+1. Each shot is generated at the next valid length and trimmed back.": ["Contagens de frames aceitas pelo modelo. H3: 17n+5; LTX/Wan: 8n+1; Wan: 4n+1. Cada plano é gerado no próximo comprimento válido e depois aparado.", "Cantidad de fotogramas que acepta el modelo. H3: 17n+5; LTX/Wan: 8n+1; Wan: 4n+1. Cada plano se genera con la siguiente duración válida y luego se recorta.", "模型接受的帧数规则。H3：17n+5；LTX/Wan：8n+1；Wan：4n+1。每个镜头按下一个有效长度生成后再裁短。"],
  "Longest shot sent to the model. Keep it within what the model / LoRA was trained on (H3 body swap: about 4.5 s).": ["Duração máxima enviada ao modelo. Mantenha dentro do que o modelo/LoRA aprendeu (body swap H3: cerca de 4,5 s).", "Duración máxima enviada al modelo. Respeta lo usado en el entrenamiento del modelo/LoRA (body swap H3: unos 4,5 s).", "发送给模型的镜头最长时长。应符合模型/LoRA 的训练长度（H3 换人约 4.5 秒）。"],
  "Shorter shots merge into a neighbour when the merge still fits the maximum.": ["Planos mais curtos são unidos ao vizinho quando a junção ainda respeita o máximo.", "Los planos más cortos se unen al vecino si la unión respeta el máximo.", "较短镜头会在合并后仍未超过上限时与相邻镜头合并。"],
  "The planner works on this frame rate (the source is resampled). Generated video and audio use it too.": ["O planner usa esta taxa de frames (o vídeo de origem é reamostrado). O vídeo e áudio gerados também a usam.", "El planificador usa esta frecuencia de fotogramas (el origen se remuestrea). El vídeo y el audio generados también la usan.", "规划器使用此帧率（源视频会重新采样）。生成的视频和音频也会使用此帧率。"],
  "Only the first N shots (0 = all). Handy for quick tests.": ["Somente os primeiros N planos (0 = todos). Útil para testes rápidos.", "Solo los primeros N planos (0 = todos). Útil para pruebas rápidas.", "仅处理前 N 个镜头（0 = 全部）。适合快速测试。"],
  "Only the first N seconds of the video (0 = all).": ["Somente os primeiros N segundos do vídeo (0 = todos).", "Solo los primeros N segundos del vídeo (0 = todos).", "仅处理视频前 N 秒（0 = 全部）。"],
  "Generation size: the source aspect ratio at this pixel area (0.15 ≈ 512×288). Cropped shots use it for the crop.": ["Tamanho da geração: mantém a proporção do vídeo nessa área de pixels (0,15 ≈ 512×288). Planos recortados usam esse valor no recorte.", "Tamaño de generación: mantiene la proporción de origen con esta área de píxeles (0,15 ≈ 512×288). Los planos recortados usan este valor.", "生成尺寸：按此像素面积保持源视频宽高比（0.15 ≈ 512×288）。裁剪镜头使用此尺寸。"],
  "Width and height snap to this multiple (32 for H3 and most video models).": ["Largura e altura são ajustadas a este múltiplo (32 para H3 e a maioria dos modelos de vídeo).", "El ancho y alto se ajustan a este múltiplo (32 para H3 y la mayoría de modelos de vídeo).", "宽度和高度会对齐到此倍数（H3 和多数视频模型使用 32）。"],
  "Auto loop: every shot in one queue run. Queue loop: one shot per run, stored on disk, the join outputs the video after the last shot.": ["Loop automático: todos os planos em uma execução da fila. Loop em fila: um plano por execução, salvo em disco; o join gera o vídeo após o último plano.", "Bucle automático: todos los planos en una ejecución. Bucle en cola: un plano por ejecución, guardado en disco; el join genera el vídeo al terminar el último.", "自动循环：一次队列运行所有镜头。队列循环：每次运行一个镜头并保存到磁盘；最后一个镜头完成后由 Join 输出视频。"],
  "Shots that do not run: keep the original video there, or remove them (with their audio).": ["Planos não executados: manter o vídeo original nesses trechos ou removê-los junto com o áudio.", "Planos que no se ejecutan: conserva el vídeo original en ese tramo o elimínalo junto con su audio.", "未运行的镜头：保留对应原视频片段，或连同音频一起移除。"],
  "Context kept around the mask's box, as a % of the box. More context blends better; less gives the model more pixels on the subject.": ["Contexto mantido ao redor da caixa da máscara, como porcentagem da caixa. Mais contexto mistura melhor; menos dá mais pixels ao sujeito.", "Contexto alrededor del cuadro de la máscara, como porcentaje. Más contexto mezcla mejor; menos ofrece más píxeles al sujeto.", "蒙版边框周围保留的上下文比例。上下文越多越容易融合；越少则主体像素越多。"],
  "Grow the mask by this many pixels: before pasting a cropped result back, and for the generation mask (inpaint 'only the mask'). Raise it when the new person is bigger than the old one (longer hair, wider body).": ["Expande a máscara por esta quantidade de pixels antes de colar o recorte e na máscara de geração. Aumente se a nova pessoa for maior (cabelo mais comprido, corpo mais largo).", "Amplía la máscara esta cantidad de píxeles antes de pegar el recorte y para la máscara de generación. Auméntala si la persona nueva es más grande.", "粘贴裁剪结果前以及生成蒙版时，将蒙版扩展指定像素。新人物更大时（头发更长、身体更宽）可调高。"],
  "Soft edge of the paste-back, in pixels.": ["Suavização da borda ao colar de volta, em pixels.", "Suavizado del borde al pegar de vuelta, en píxeles.", "贴回时边缘的柔化像素数。"],
  "Hold the mask this many frames before and after: removes flicker where SAM 3 drops a frame.": ["Estende a máscara por esta quantidade de frames antes e depois; reduz falhas quando o SAM 3 perde um frame.", "Mantiene la máscara estos fotogramas antes y después; reduce parpadeos cuando SAM 3 pierde uno.", "蒙版前后延续指定帧数，减少 SAM 3 漏掉某帧时的闪烁。"],
  "Square blocks of this size (0 = off). 16 matches H3's latent grid, useful as an inpainting mask.": ["Blocos quadrados deste tamanho (0 = desligado). 16 corresponde à grade latente do H3 e pode ser útil como máscara de inpainting.", "Bloques cuadrados de este tamaño (0 = desactivado). 16 coincide con la cuadrícula latente de H3 y sirve para inpainting.", "使用此尺寸的方块（0 = 关闭）。16 与 H3 潜空间网格匹配，可用于修补蒙版。"],
  "SAM 3 score needed to keep a text match. Lower finds more (and more wrong) objects.": ["Pontuação mínima do SAM 3 para manter uma correspondência por texto. Valores menores encontram mais objetos, inclusive incorretos.", "Puntuación mínima de SAM 3 para conservar una coincidencia de texto. Un valor menor encuentra más objetos, también erróneos.", "保留文本匹配所需的 SAM 3 分数。调低会找到更多对象，也会增加误检。"],
  "How many matches of the text prompt are tracked together.": ["Quantas correspondências do texto serão rastreadas juntas.", "Cuántas coincidencias del texto se siguen a la vez.", "同时跟踪多少个符合文本的对象。"],
  "Only the mask: the result replaces the masked pixels. The whole box: the full crop is pasted with feathered borders.": ["Somente máscara: o resultado substitui os pixels mascarados. Caixa inteira: todo o recorte é colado com bordas suaves.", "Solo máscara: el resultado sustituye los píxeles enmascarados. Cuadro completo: se pega todo el recorte con bordes suavizados.", "仅蒙版：结果替换蒙版像素。整个框：粘贴完整裁剪区域并柔化边缘。"],
  "How many frames of each shot the VLM looks at.": ["Quantos frames de cada plano o VLM analisa.", "Cuántos fotogramas de cada plano analiza el VLM.", "VLM 分析每个镜头的帧数。"], "Longest VLM answer. Raise it for long descriptions.": ["Tamanho máximo da resposta do VLM. Aumente para descrições longas.", "Longitud máxima de respuesta del VLM. Auméntala para descripciones largas.", "VLM 回复的最大长度。长描述可调高。"],
};
export function t(value) {
  const s = String(value ?? ""), lang = uiLanguage.value;
  if (lang === "en") return s;
  const entry = TEXT[s];
  if (entry) return entry[{ pt: 0, es: 1, zh: 2 }[lang]];
  let m;
  if ((m = s.match(/^Person (\d+)$/))) return `${t("Person")} ${m[1]}`;
  if ((m = s.match(/^Shot #(\d+)( · points)?$/))) return `${t("Shot")} #${m[1]}${m[2] ? ` · ${lang === "pt" ? "pontos" : lang === "es" ? "puntos" : "点"}` : ""}`;
  if ((m = s.match(/^(\d+) people · (\d+) linked$/))) return lang === "pt" ? `${m[1]} pessoas · ${m[2]} vinculadas` : lang === "es" ? `${m[1]} personas · ${m[2]} vinculadas` : `${m[1]} 人物 · 已关联 ${m[2]} 人`;
  if ((m = s.match(/^(\d+) shot\(s\) will run$/))) return lang === "pt" ? `${m[1]} plano(s) serão executados` : lang === "es" ? `se ejecutarán ${m[1]} plano(s)` : `将运行 ${m[1]} 个镜头`;
  if ((m = s.match(/^(\d+) suggestions$/))) return lang === "pt" ? `${m[1]} sugestões` : lang === "es" ? `${m[1]} sugerencias` : `${m[1]} 条建议`;
  if ((m = s.match(/^(\d+) shot\(s\)$/))) return lang === "pt" ? `${m[1]} plano(s)` : lang === "es" ? `${m[1]} plano(s)` : `${m[1]} 个镜头`;
  if ((m = s.match(/^(\d+) warning\(s\) · (\d+) note\(s\)$/))) return lang === "pt" ? `${m[1]} aviso(s) · ${m[2]} observação(ões)` : lang === "es" ? `${m[1]} aviso(s) · ${m[2]} nota(s)` : `${m[1]} 条警告 · ${m[2]} 条备注`;
  return s;
}
function localizeProps(props) {
  if (!props || typeof props !== "object") return props;
  let changed = false; const out = { ...props };
  for (const k of ["title", "placeholder", "aria-label"]) if (typeof out[k] === "string") { const v = t(out[k]); changed ||= v !== out[k]; out[k] = v; }
  return changed ? out : props;
}
function localizeChildren(value) {
  if (typeof value === "string") return t(value);
  if (Array.isArray(value)) return value.map(localizeChildren);
  return value;
}
// All view modules use this h wrapper, so labels, placeholders, tooltips and section titles stay in sync.
export function h(type, props, children) {
  const hasChildrenArg = arguments.length >= 3;
  if (!hasChildrenArg && props?.__v_isVNode) return vueH(type, null, props);
  const p = localizeProps(props);
  if (hasChildrenArg) return vueH(type, p, localizeChildren(children));
  if (typeof props === "string" || Array.isArray(props)) return vueH(type, localizeChildren(props));
  return vueH(type, p);
}

export const GRIDS = { "H3 (17n+5)": [17, 5], "LTX / Wan (8n+1)": [8, 1], "Wan (4n+1)": [4, 1], "any": [1, 0] };
export const DEFAULTS = {
  video: "", fps: 24, grid: "H3 (17n+5)", mode: "shots", max_s: 4.5, min_s: 1.0, sensitivity: 0.5,
  max_parts: 0, max_total_s: 0, start_s: 0, end_s: 0, start_f: null, end_f: null, track: null, bounds: [], segs: [], global_ref: "", global_ref2: "", global_prompt: "", mask_video: "",
  megapixels: 0.5, multiple: 32, detector: "adaptive", run: "auto", auto_continue: true,
  skip_fill: "original", audio_mode: "auto", ref_details: {}, mask_cfg: {}, vlm_cfg: {}, cast: {}, cast_assign: true, cast_split: false, cast_only: false,
  filters: { person: false, min_person_area: 0, max_persons: 0, face: false, skip_dark: false, dark_level: 0.06,
             skip_static: false, static_level: 0.004, min_frames: 0, samples: 6 },
};

// global mask settings (same defaults as the server's DEFAULT_MASK)
export const MASK_DEFAULTS = { threshold: 0.5, max_objects: 4, invert: false, fill_holes: true, temporal_expand: 2, blockify: 0,
                               padding: 0.15, expand: 16, feather: 12, paste: "mask" };

export const VLM_DEFAULTS = { enabled: false, frames: 3, max_tokens: 1024, auto_segment: true, auto_shot: true, instruction: "",
                              describe_preset: "short", describe_custom: "", fit_prompt: true, temperature: 0, seed: 0 };

// hover help for every setting (the 📖 Guide button opens the full documentation)
export const TIPS = {
  "Mode": "Camera cuts: shots start at the detected cuts, long ones are split evenly. Fixed length: equal parts no longer than the maximum.",
  "Detector": "Cut detector. PySceneDetect adaptive works on most footage (fast camera moves don't trigger it); content is stricter on hard cuts; built-in needs no package.",
  "Sensitivity": "How easily a change counts as a cut. Higher finds more cuts (and more false ones).",
  "Frame grid": "Frame counts the model accepts. H3: 17n+5, LTX/Wan: 8n+1, Wan: 4n+1. Each shot is generated at the next valid length and trimmed back.",
  "Max seconds / shot": "Longest shot sent to the model. Keep it within what the model / LoRA was trained on (H3 body swap: about 4.5 s).",
  "Min seconds / shot": "Shorter shots merge into a neighbour when the merge still fits the maximum.",
  "Timeline fps": "The planner works on this frame rate (the source is resampled). Generated video and audio use it too.",
  "Max shots": "Only the first N shots (0 = all). Handy for quick tests.",
  "Megapixels": "Generation size: the source aspect ratio at this pixel area (0.5 ≈ 960×544 for 16:9). Cropped shots use it for the crop.",
  "Size multiple": "Width and height snap to this multiple (32 for H3 and most video models).",
  "Run": "Auto loop: every shot in one queue run. Queue loop: one shot per run, stored on disk, the join outputs the video after the last shot.",
  "Skipped shots in the output": "Shots that do not run: keep the original video there, or remove them (with their audio).",
  "Crop padding": "Context kept around the mask's box, as a % of the box. More context blends better; less gives the model more pixels on the subject.",
  "Expand": "Grow the mask by this many pixels: before pasting a cropped result back, and for the generation mask (inpaint 'only the mask'). Raise it when the new person is bigger than the old one (longer hair, wider body).",
  "Feather": "Soft edge of the paste-back, in pixels.",
  "Temporal expand": "Hold the mask this many frames before and after: removes flicker where SAM 3 drops a frame.",
  "Blockify": "Square blocks of this size (0 = off). 16 matches H3's latent grid, useful as an inpainting mask.",
  "Detection threshold": "SAM 3 score needed to keep a text match. Lower finds more (and more wrong) objects.",
  "Max objects": "How many matches of the text prompt are tracked together.",
  "Paste back": "Only the mask: the result replaces the masked pixels. The whole box: the full crop is pasted with feathered borders.",
  "Frames per shot": "How many frames of each shot the VLM looks at.",
  "Max tokens": "Longest VLM answer. Raise it for long descriptions.",
};

export const snapUp = (n, grid) => {
  const [s, o] = GRIDS[grid] || GRIDS["H3 (17n+5)"]; n = Math.max(1, n | 0);
  if (s === 1) return n; if (n <= o) return o; return o + s * Math.ceil((n - o) / s);
};
export const snapDown = (n, grid) => {
  const [s, o] = GRIDS[grid] || GRIDS["H3 (17n+5)"]; n = Math.max(1, n | 0);
  if (s === 1) return n; if (n <= o) return o; return o + s * Math.floor((n - o) / s);
};
export const hue = i => `hsl(${(i * 47 + 200) % 360} 55% 46%)`;
export const fmtT = (f, fps) => { const s = f / fps; return `${Math.floor(s / 60)}:${(s % 60).toFixed(2).padStart(5, "0")}`; };
export const segKey = s => `${s.start}-${s.end}`;
// a shot has a mask: its own mask video, SAM 3 text or points, or the plan's mask video for the whole video
// (the planner's `mask` input is only known when the workflow runs)
export const hasMask = s => !!(s.mask?.video || s.mask?.text || (s.mask?.points || []).length || s.extMask);
export const maskSource = s => s.mask?.video ? "video" : (s.mask?.text || (s.mask?.points || []).length) ? "sam"
  : s.trackMask ? "track" : s.extMask ? "global" : "";

// ---- small view helpers
export const pill = (text, kind = "", title = "") => h("span", { class: ["pill", kind], title }, text);
export const hint = (text, style = "") => h("span", { class: "hint", style }, text);
export const fld = (label, input, help) => {
  const tip = Object.entries(TIPS).find(([k]) => String(label).startsWith(k))?.[1] || "";
  return h("div", { class: "fld", title: tip }, [h("label", tip ? [label, h("span", { class: "qm" }, " ⓘ")] : label), input,
    help ? h("div", { class: "hint" }, help) : null]);
};
export const check = (checked, onChange, label, title = "") =>
  h("label", { class: "chk", title }, [h("input", { type: "checkbox", checked: !!checked, onChange: e => onChange(e.target.checked) }), label]);
export const select = (value, options, onChange, attrs = {}) => h("select", { value, onChange: e => onChange(e.target.value), ...attrs },
  options.map(o => h("option", { value: Array.isArray(o) ? o[0] : o }, Array.isArray(o) ? o[1] : o)));
// a titled section inside a card; `open` makes it a collapsible <details>
// what the user opened or closed wins over a section's default: re-renders (a timer, a status line) no longer snap a
// section back shut right after it was clicked open
const OPEN = reactive({});
export const section = (title, body, { sub = "", right = null, open = null, cls = "" } = {}) => {
  const head = [h("span", { class: "sect" }, title), sub ? hint(sub) : null, h("span", { class: "grow" }), right];
  if (open === null) return h("div", { class: ["sec", cls] }, [h("div", { class: "sech" }, head), ...[body].flat()]);
  const key = `${cls}|${typeof title === "string" ? title : ""}`;
  const isOpen = key in OPEN ? OPEN[key] : !!open;
  return h("details", { class: ["sec", cls], open: isOpen, onToggle: e => { OPEN[key] = e.target.open; } },
    [h("summary", { class: "sech" }, [h("span", { class: "caret" }, "▸"), ...head]), ...[body].flat()]);
};
