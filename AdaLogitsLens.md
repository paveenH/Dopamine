## **Top-k Visualization**

**Llama3-8B-IT**

![image.png](attachment:02f17ae7-76a3-4d66-a0ca-d67da93f3d49:image.png)

![image.png](attachment:db031dde-5d35-4ff0-a8ec-f4deacf744e4:image.png)

## Residual Stream Alignment

这种现象在 Transformer 解释性研究中通常被称为 **"Privileged Basis"** 或 **"Residual Stream Alignment"**。

- **残差流假说 (The Highway Hypothesis)：**Transformer 的 Residual Stream 就像一条高速公路。如果模型觉得“信心”这个信号极其重要，它可能会在高速公路上专门划出一条“专用车道”（即潜空间中的一个固定方向）。
    - 由于 Residual Connections 的存在，Layer $L$ 输出的信号会直接加到 Layer $L+1$ 上。
    - 垂直条纹的解释： 那些垂直对齐的神经元，很可能是“读写同一条专用车道”的节点。这种机制被称为 "Signal Amplification"或 "Maintenance"。
    - 从 "Communication" 转向 "Maintenance"：**Signal Maintenance or** **Drift Prevention**
        - Transformer 的 Residual Stream 充满了噪声和其他 Head 写入的干扰信息。如果“专家人设”只在第 10 层写入一次，到了第 25 层可能就被 LayerNorm 和其他注意力头“冲刷”掉了。
        - Index 4055 在 Layer 5, 6...15, 16... 反复出现。这意味着模型在主动、持续地给“自信/结构”这个信号**“充电”**。它在每一层都重新确认：“保持代码格式，压制犹豫”。→ 这证明了 Role 并不是一个一次性的 Switch，而是一个需要持续维护的动态过程 (Active Maintenance Process)。
- **Paper**: "[Privileged Bases in the Transformer Residual Stream](https://transformer-circuits.pub/2023/privileged-basis/index.html)" (Anthropic, 2023)
    - 核心观点： 在残差流中，并非所有方向都是等价的。有些特定的方向（即神经元基底）是“特权”的，模型倾向于用这些特定的神经元来编码关键特征，而不是随机旋转的方向。
    - 如果基底不是特权的，你的 RSN 应该是每一层都旋转到不同的 Index；正是因为存在 Privileged Basis，你才能观察到垂直对齐的 Index 4055。
- **Paper**: "[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)" (Anthropic, 2021)
    - 核心观点： 正式定义了 Residual Stream as a Communication Channel 解释了层与层之间是如何通过读写残差流的子空间（Subspaces）来传输信息的。
    - 对你的用处： 这是“高速公路假说”的鼻祖文献。你可以引用它来定义你的“Role Subspace”。
- **Paper:** "[Transformer Dynamics: A neuroscientific approach to interpretability of large language models](https://arxiv.org/abs/2502.12131)" (arXiv, 2025)
    - 核心观点： 将残差流视为一个随层数演变的动态系统。研究发现，尽管存在非线性干扰，单个残差流单元（Individual Units）在跨层时表现出惊人的强连续性（Strong Continuity）。
    - 直接印证了你观察到的“垂直条纹”。你可以引用它说：你的发现与最新的“Transformer Dynamics”研究一致，证实了 Role 信号也是通过这种跨层连续性来维持的。

## Results

### Logit Lens

| **Index** | **Mark** | Note | **Tokens +**  | **reversal Tokens -** |
| --- | --- | --- | --- | --- |
| **2629** | None | reversal 也多为乱码/不可分类 | 'NCY'(0.08), 'bine'(0.08), '
**4,5** | ' **Uncategorized**'(0.03), 'ří'(0.03), '**ROID**'(0.03), 'mith'(0.03), 'illing'(0.03)
**14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31** |
| **2692** | None | reversal 是 function words | ' eskort'(0.21), 'lacak'(0.20), ' erotik'(0.20), ' seksi'(0.18), 'alars'(0.18) | ' ac'(0.05), ' ad'(0.04), ' a'(0.04), ' ve'(0.04), ' yet'(0.04)
**1,2,3,4,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31** |
| **1731** | CODE | Unnamed, .Companion, .gridColumn | 'Unnamed'(0.07), 'totals'(0.07), '.Companion'(0.07), '.gridColumn'(0.06), 'Â'(0.06)
**10,11,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28** | 'Unnamed'(0.02), 'totals'(0.02), '.Companion'(0.02), '.gridColumn'(0.02), 'Â'(0.02) |
| **4055** | CODE | #af, #ad, /******/ → 明显代码注释 | '#af'(0.11), '/******/'(0.09), '#ad'(0.09), '#ab'(0.09), ' -*-
\\n'(0.09)
**5,6,9,11** | '#af'(0.05), '/******/'(0.04), '#ad'(0.04), '#ab'(0.04), ' -*-
\\n'(0.04)
**1,7,14,15,16,17,18,20,21,22,23,24** |
| **373** | Structural |  | ' '(0.09), '/Instruction'(0.07), '**<|end_of_text|>**'(0.07), '  '(0.06), 'oha'(0.06) | 'Ư'(0.06), ' Dün'(0.05), '/DTD'(0.05), 'iêu'(0.05), 'imiter'(0.05)
**9,18,19,20,21,22,23,24,25,26,27,28,29,30,31** |
| **3585** | NONE | itself, also, latter（普通词）不足以构成语义方向 | ' itself'(0.07), 'ehr'(0.07), ' also'(0.07), '..\\n'(0.07), ' latter'(0.07)
**15,16,17,18,19,20,21,22,23,24,25,26,27,28** | ' itself'(0.02), 'ehr'(0.02), ' also'(0.02), '..\\n'(0.02), ' latter'(0.02) |
| **3070** | NONE | ¼, ¾, afc（无语义 cluster） | '¼'(0.13), '¾'(0.13), 'afc'(0.12), 'afb'(0.12), 'afd'(0.11) | '��'(0.02), '��'(0.02), '��'(0.02), ''(0.02), 'ーリ'(0.02)
**5,6,7,8,9,18,19,20,21,22,23,24,25,26** |
| **133** | **HES
引入思考** | **reversal 是 …but, …I, …and** | **' “'(0.11), ' (“'(0.10), '’\\n'(0.09), ' ‘'(0.09), '—"'(0.08)** | **'…but'(0.05), '…I'(0.04), '…\\n'(0.04), '…and'(0.04), 'شمالی'(0.04)
5,6,18,19,20,21,22,23,24,25,26,27,30,31** |
| **873** | NONE | buflen, inFile, quartered（文件/处理） | ' buflen'(0.08), 'quartered'(0.07), ' іншого'(0.07), ' inFile'(0.07), '++++++++++++++++'(0.07) | 'sel'(0.03), 'واء'(0.03), 'aneously'(0.03), 'infeld'(0.03), 'hesion'(0.03)
**18,19,20,21,22,23,24,25,26,27,28,29,30** |
| **1298** | NONE | $MESS, @js, lamaz, İz（非连贯 cluster） | '$MESS'(0.10), '@js'(0.08), '">×</'(0.07), 'lamaz'(0.06), ' İz'(0.06) | 'รงเร'(0.03), 'ราะ'(0.03), '็กชาย'(0.03), 'भग'(0.02), 'คโน'(0.02)
**18,19,20,22,23,24,25,26,27,28,29,30** |
| **2646** | NONE | yoluyla, aracılığıyla（特定土耳其语模式） | ' yoluyla'(0.10), ' aracılığıyla'(0.10), 'ettes'(0.10), ' Všech'(0.10), '\\_'(0.10) | 'носят'(0.02), 'нимает'(0.02), 'assen'(0.02), 'verts'(0.01), 'ovit'(0.01)
**14,15,16,17,18,19,20,21,22,23,24** |
| **1421** | **HES** | **Honestly, Certainly, Whilst, Alright → 强烈 certainty** | **'Honestly'(0.13), 'Certainly'(0.13), 'Funny'(0.12), 'Whilst'(0.11), 'Alright'(0.11)
5,6,23,24,25,26,27,28,29,30,31** | **'Honestly'(0.05), 'Certainly'(0.05), 'Funny'(0.05), 'Whilst'(0.04), 'Alright'(0.04)** |
| **2352** | CODE | .Reporting, .Dictionary, .DropDown, underlines | '_________________\\n\\n'(0.36), '.Reporting'(0.33), '.Dictionary'(0.32), '.DropDown'(0.32), '!\\n\\n\\n\\n'(0.30)
**10,12,13** | '_________________\\n\\n'(0.02), '.Reporting'(0.02), '.Dictionary'(0.02), '.**DropDown**'(0.02), '!\\n\\n\\n\\n'(0.02)
**18,20,21,22,27,29,30,31** |
| **1189** | **HES** | **…the, …and, …I, …but → 典型 hesitation 方向** | **'…the'(0.16), '…and'(0.15), '…I'(0.15), '…but'(0.14), '…it'(0.14)
22,23,24,25,26,27,28,29,30** | **'…the'(0.06), '…and'(0.06), '…I'(0.06), '…but'(0.05), '…it'(0.05)
3,6** |
| **291** | NONE | swire + Japanese さら + German BITTE → 多语言混合 | 'swire'(0.08), 'さら'(0.07), 'donnees'(0.07), ' BITTE'(0.07), ' STDCALL'(0.07) | 'ché'(0.02), ' böylece'(0.02), ' aforementioned'(0.02), '到底'(0.02), '-icons'(0.02)
**1,7,13,14,15,16,17,27,28,29** |
| **3695** | NONE | OrCreate, avier, mere → 模糊，接近函数名但不稳定 | 'OrCreate'(0.07), 'avier'(0.07), 'ksen'(0.07), 'mere'(0.06), ' cref'(0.06)
**19,20,21,22,25,26,27,28,31** | 'OrCreate'(0.03), 'avier'(0.03), 'ksen'(0.02), 'mere'(0.02), ' cref'(0.02) |
| **3516** | **CODE** | **sourceMapping, wireType, addCriterion** | **' sourceMapping'(0.10), ' wireType'(0.09), ' addCriterion'(0.08), ' останні'(0.08), ' eskort'(0.08)
1,2,3,4,6** | **' sourceMapping'(0.03), ' wireType'(0.03), ' addCriterion'(0.02), ' останні'(0.02), ' eskort'(0.02)
26,27,28,30** |
| **2932** | NONE | NavController, ViewChild, ######## → Angular-like tokens | 'NavController'(0.10), ' ########################################################################'(0.09), ' ViewChild'(0.08), 'ίγ'(0.08), 'ynos'(0.08)
**10,11** | 'NavController'(0.00), ' ########################################################################'(0.00), ' ViewChild'(0.00), 'ίγ'(0.00), 'ynos'(0.00)
**1,2,3,4,5,6,7** |
| **2184** | NONE | ataset, visor, udur（接近数据项，但不稳定） | 'ataset'(0.07), '#__'(0.07), 'visor'(0.07), 'udur'(0.06), 'otts'(0.06)
**21,24,25,26,27,28,29,30** | 'ataset'(0.02), '#__'(0.02), 'visor'(0.02), 'udur'(0.02), 'otts'(0.02) |
| **2265** | NONE | etc, /etc, #ad（类似代码，但 mixed） | ' etc'(0.10), '/etc'(0.09), ' неї'(0.09), ' нього'(0.09), '#ad'(0.09)
**11,15,16** | ' etc'(0.01), '/etc'(0.01), ' неї'(0.01), ' нього'(0.01), '#ad'(0.01)
**1,2,3** |
| **761** | NONE | strtotime, eza, arus（有主题性，但不强） | 'arus'(0.07), 'esz'(0.07), ' olmadı'(0.07), 'strtotime'(0.07), 'eza'(0.07) | 'orb'(0.02), '.**dateFormat**'(0.02), 'illez'(0.02), 'ourd'(0.01), 'itchen'(0.01)
**16,23,24,25,26,27** |
| **2082** | LANG | イク（日文片假），lon, CAA（非语义 cluster） | 'イク'(0.08), 'lon'(0.08), 'ewith'(0.08), 'CAA'(0.08), ' fi'(0.08) | 'enco'(0.00), 'ansi'(0.00), 'ulas'(0.00), 'engl'(0.00), 'abad'(0.00)
**3,4,5,6,10,11** |
| **384** | NONE | stad, DEV, usz, IDADE（类似命名空间） | 'stad'(0.07), 'DEV'(0.06), 'usz'(0.06), 'IDADE'(0.06), 'ider'(0.06)
**18,19,21,22,23,24** | 'stad'(0.01), 'DEV'(0.01), 'usz'(0.01), 'IDADE'(0.01), 'ider'(0.01) |
| **1130** | NONE | _tF, _tA, .pub, 英雄 → 混合类别 | '_tF'(0.07), '_tA'(0.06), 'inary'(0.06), '.pub'(0.06), '英雄'(0.06)
**17,18,19,20,21** | '_tF'(0.01), '_tA'(0.01), 'inary'(0.01), '.pub'(0.01), '英雄'(0.01) |
| **2977** | LANG | 项, 파일첨부, 新글, …（中文、韩文夹杂） | ' 项'(0.07), ' 파일첨부'(0.06), '\\n'(0.06), ' 새글'(0.06), ' '(0.06) | '.ISupportInitialize'(0.00), '。www'(0.00), 'надлеж'(0.00), '駅徒歩'(0.00), '№№'(0.00)
2,3,4,5,6 |
| **1122** | NONE | ×, fold, NewLabel, né → GUI-ish，但不构成方向 | '">×</'(0.10), 'fold'(0.09), 'ร'(0.09), 'NewLabel'(0.07), 'né'(0.07) | '">×</'(0.03), 'fold'(0.03), 'ร'(0.03), 'NewLabel'(0.02), 'né'(0.02)
**19,20,23,24,30** |
| **2303** | CODE | …\n, …\n, “ ”, … → 典型 ellipsis family | '…\\n'(0.13), '...\\n'(0.12), ' '(0.12), '…'(0.12), '\\n'(0.12)
**3,4,5,6,7** | '…\\n'(0.00), '...\\n'(0.00), ' '(0.00), '…'(0.00), '\\n'(0.00) |
| **3266** | NONE | 이는, žal, fak, !. → 部分韩文 + 情绪词（mix），但 reversal 有 ellipsis | ' 이는'(0.08), 'aan'(0.06), ' žal'(0.06), 'fak'(0.06), '!.'(0.06) | ' …\\n'(0.01), '…\\n'(0.01), ' unmistak'(0.01), ' […]\\n'(0.01), ' aforementioned'(0.01)
**1,14,15,16,17** |
| **281** | CODE | .SizeType, .Undef, .Guna, .SetParent | '.SizeType'(0.10), '.Undef'(0.08), '.Guna'(0.08), '.SetParent'(0.07), '.Cursors'(0.06)
**22,23,24,25,26** | '.SizeType'(0.02), '.Undef'(0.02), '.Guna'(0.02), '.SetParent'(0.02), '.Cursors'(0.01) |
- **HES** = Hesitation / Certainty / 语气方向
- **CODE** = Code / formatting / markup
- **LANG** = 多语言/Unicode cluster（非语义）
- **TOPIC** = 主题相关（特定 domain，如色情、文件处理、dataset）
- **NONE** = 不足以判断 / 不明显

| **Layer_Index** | **Top_Positive_Tokens (Expert Adds)** | **Top_Negative_Tokens (Expert Removes/Non-Exp Adds)** |
| --- | --- | --- |
| **1** | 'λω'(0.00), ' loose'(0.00), ' Loose'(0.00), 'ocab'(0.00), '244'(0.00), 'acher'(0.00), 'ukt'(0.00), '877'(0.00), 'oled'(0.00), 'MOOTH'(0.00) | 'esan'(-0.00), 'tfoot'(-0.00), 'inta'(-0.00), 'omik'(-0.00), 'arry'(-0.00), 'oug'(-0.00), 'onica'(-0.00), 'REET'(-0.00), 'kses'(-0.00), '�'(-0.00) |
| **2** | 'icrous'(0.00), 'ocab'(0.00), 'ertools'(0.00), 'ród'(0.00), '_vlog'(0.00), 'phies'(0.00), 'izoph'(0.00), '_DECREF'(0.00), '_Tis'(0.00), '데이트'(0.00) | 'ility'(-0.00), 'o'(-0.00), '.dd'(-0.00), 'v'(-0.00), 'esan'(-0.00), 'ech'(-0.00), '-'(-0.00), 'chten'(-0.00), 'cheng'(-0.00), 'Original'(-0.00) |
| **3** | 'acemark'(0.00), 'ycop'(0.00), 'errick'(0.00), '(火'(0.00), '/copyleft'(0.00), 'ertools'(0.00), 'RunWith'(0.00), 'PropertyChanged'(0.00), 'reet'(0.00), ' зависим'(0.00) | 'ahren'(-0.00), 'chan'(-0.00), 'cheng'(-0.00), 'erdem'(-0.00), 'yw'(-0.00), ' eth'(-0.00), 'edar'(-0.00), 'PLICIT'(-0.00), '.trailing'(-0.00), 'аз'(-0.00) |
| **4** | 'teş'(0.01), 'Архів'(0.01), 'ioctl'(0.01), 'TouchUpInside'(0.00), 'ینک'(0.00), 'unte'(0.00), 'emonic'(0.00), ' Hamm'(0.00), 'ember'(0.00), '"">//'(0.00) | 'apon'(-0.01), 'rani'(-0.01), ' '(-0.00), 'wor'(-0.00), 'ritt'(-0.00), 'hire'(-0.00), 'hi'(-0.00), 'hle'(-0.00), '.tb'(-0.00), 'Pot'(-0.00) |
| **5** | 'clist'(0.01), 'PageRoute'(0.01), ' Hamm'(0.01), 'enville'(0.01), 'nel'(0.01), 'ku'(0.01), ' tumble'(0.01), ' últ'(0.01), 'ac'(0.01), 'inking'(0.01) | 'anax'(-0.01), 'nard'(-0.01), 'ENCIL'(-0.01), 'ити'(-0.01), 'omik'(-0.01), 'apus'(-0.01), 'ukan'(-0.01), 'iti'(-0.01), 'omu'(-0.01), '.gdx'(-0.01) |
| **6** | 'undra'(0.01), 'pei'(0.01), 'rees'(0.01), ' Hra'(0.01), 'änn'(0.01), 'STYPE'(0.01), ' inne'(0.01), 'PHA'(0.01), 'orph'(0.01), 'bia'(0.01) | 'anson'(-0.01), 'itele'(-0.01), ' pri'(-0.01), ' Energ'(-0.01), 'umed'(-0.01), 'akis'(-0.01), 'パン'(-0.01), 'clipse'(-0.01), 'id'(-0.01), 'rez'(-0.01) |
| **7** | 'Skip'(0.02), 'dol'(0.02), 'orp'(0.01), 'ven'(0.01), ' Stokes'(0.01), 'leen'(0.01), 'blocking'(0.01), 'alığı'(0.01), 'ướ'(0.01), 'ưỡng'(0.01) | 'ід'(-0.02), 'プレ'(-0.02), 'tridge'(-0.02), '796'(-0.02), '�'(-0.02), 'anvas'(-0.02), 'itele'(-0.02), ' princip'(-0.01), 'eyse'(-0.01), 'iefs'(-0.01) |
| **8** | '械'(0.03), 'ernals'(0.03), 'nj'(0.02), ' shoe'(0.02), 'وره'(0.02), 'ordum'(0.02), 'arkin'(0.02), 'umlu'(0.02), 'wayne'(0.02), ' stoi'(0.02) | 'oyo'(-0.03), 'ulong'(-0.02), ' Mos'(-0.02), 'uder'(-0.02), 'ExecutionContext'(-0.02), 'ovit'(-0.02), 'リーズ'(-0.02), ' bedrooms'(-0.02), ' Mam'(-0.02), ' swallow'(-0.02) |
| **9** | 'utenberg'(0.03), '.GetService'(0.03), ' Vám'(0.03), 'κέ'(0.03), '//{{'(0.03), 'anja'(0.03), '■■'(0.03), 'âng'(0.03), 'extField'(0.03), 'ernals'(0.03) | 'isty'(-0.03), 'oa'(-0.03), 'ities'(-0.03), 'kip'(-0.03), 'aeper'(-0.03), 'opot'(-0.02), ' Hamp'(-0.02), 'ille'(-0.02), ' Eudicots'(-0.02), ' Daw'(-0.02) |
| **10** | 'dü'(0.05), '//{{'(0.04), '誠'(0.04), '.SetParent'(0.04), 'ď'(0.04), 'oloj'(0.04), '.GetService'(0.04), '×\\n\\n'(0.04), '.opensource'(0.04), 'っと'(0.04) | ' '(-0.03), 'l'(-0.03), ' inv'(-0.03), ' gr'(-0.03), ' Hugh'(-0.03), 'uya'(-0.03), ' Gent'(-0.03), ' boy'(-0.03), ' ground'(-0.03), ' i'(-0.03) |
| **11** | 'ikip'(0.05), 'PECT'(0.05), '_dual'(0.05), 'ď'(0.05), 'pector'(0.04), 'adium'(0.04), '�'(0.04), 'ainen'(0.04), '.cloudflare'(0.04), 'ommen'(0.04) | 'oster'(-0.04), 'ovich'(-0.04), 'uya'(-0.04), ' Gerr'(-0.03), ' stretch'(-0.03), ' sh'(-0.03), 'ETS'(-0.03), ' soak'(-0.03), ' probable'(-0.03), 'iste'(-0.03) |
| **12** | 'aint'(0.07), ' тро'(0.06), 'aur'(0.06), 'مال'(0.06), 'ikt'(0.06), 'mal'(0.06), 'rios'(0.06), 'sel'(0.06), 'zos'(0.06), 'ؤ'(0.06) | 'boys'(-0.06), '444'(-0.06), 'boy'(-0.06), '952'(-0.06), 'ovich'(-0.06), 'ESCO'(-0.06), 'aggi'(-0.06), 'ковий'(-0.06), 'ardless'(-0.06), '/socket'(-0.05) |
| **13** | 'aines'(0.08), 'acker'(0.07), '#ad'(0.07), 'śnie'(0.06), 'mal'(0.06), 'klä'(0.06), 'ergus'(0.06), 'uds'(0.06), 'idia'(0.06), 'swer'(0.06) | 'uese'(-0.06), 'eldon'(-0.06), 'pta'(-0.06), ' bab'(-0.06), 'igg'(-0.06), 'hausen'(-0.06), '�'(-0.06), '591'(-0.06), '556'(-0.06), '606'(-0.06) |
| **14** | 'acker'(0.07), 'mal'(0.07), 'OutOfRange'(0.07), ' link'(0.07), 'nak'(0.07), 'ICODE'(0.07), '�'(0.07), 'ugi'(0.06), ' correct'(0.06), 'idia'(0.06) | 'ôn'(-0.08), 'ẽ'(-0.08), 'erman'(-0.08), 'ndo'(-0.08), 'opoulos'(-0.08), 'ril'(-0.08), '_VARIABLE'(-0.07), '.Module'(-0.07), 'پر'(-0.07), 'lsen'(-0.07) |
| **15** | 'uč'(0.08), ' frags'(0.08), 'ateg'(0.08), 'الا'(0.08), 'ATES'(0.08), ' rang'(0.07), 'tuk'(0.07), '_singular'(0.07), 'ско'(0.07), 'undry'(0.07) | '狐'(-0.09), '977'(-0.09), 'ğit'(-0.08), 'lech'(-0.08), 'iare'(-0.08), '996'(-0.08), ' دام'(-0.08), 'opoulos'(-0.08), 'quir'(-0.08), 'aggio'(-0.08) |
| **16** | ' imper'(0.09), 'ateg'(0.08), 'urd'(0.08), '특별'(0.08), ' Vig'(0.08), 'th'(0.08), ' semiclass'(0.07), ' apl'(0.07), 'gear'(0.07), '**notated**'(0.07) | 'rike'(-0.11), 'วก'(-0.10), 'ibling'(-0.10), 'AMIL'(-0.10), 'landing'(-0.09), 'ilion'(-0.09), 'öße'(-0.09), '備'(-0.09), 'DTD'(-0.09), 'ğit'(-0.09) |
| **17** | 'ateg'(0.10), ' **finally**'(0.09), 'aca'(0.09), 'ipment'(0.09), ' '(0.09), ' Anders'(0.09), 'anner'(0.08), ' urn'(0.08), ' escal'(0.08), ' buffs'(0.08) | 'วก'(-0.12), ' हव'(-0.11), ' dna'(-0.11), 'ắm'(-0.11), '.**DropDown**'(-0.11), '.**ErrorMessage**'(-0.11), 'fty'(-0.11), '眉'(-0.10), '.hasMore'(-0.10), 'azer'(-0.10) |
| **18** | ' '(0.16), '1'(0.13), 'aca'(0.12), 'ipment'(0.11), 'ronics'(0.11), '2'(0.10), ' aqu'(0.10), ' **thoroughly**'(0.10), ' **substantial**'(0.10), 'okens'(0.10) | '.kotlin'(-0.19), '.**DropDown**'(-0.19), ' ruk'(-0.17), ' elektrik'(-0.16), 'zan'(-0.16), 'etine'(-0.16), '.ends'(-0.16), '.myapplication'(-0.16), ' округ'(-0.16), 'ěř'(-0.15) |
| **19** | '1'(0.16), ' '(0.14), '2'(0.14), '4'(0.13), 'A'(0.13), 'aca'(0.13), ' A'(0.12), '3'(0.12), 'ronics'(0.12), 'ipment'(0.12) | '.kotlin'(-0.18), 'iš'(-0.18), 'berman'(-0.17), ' عش'(-0.17), ' округ'(-0.17), 'PELL'(-0.16), 'хід'(-0.16), 'bdb'(-0.16), 'bris'(-0.16), ' 외'(-0.16) |
| **20** | '**A**'(0.15), 'ronics'(0.14), ' A'(0.13), '1'(0.13), ' '(0.12), 'altung'(0.12), ' vital'(0.12), ' Vig'(0.12), ' simply'(0.12), 'clar'(0.12) | '×\\n\\n'(-0.19), '.**DropDown**'(-0.19), 'ERNEL'(-0.19), 'ěř'(-0.18), 'ooke'(-0.18), 'iš'(-0.18), '>[]'(-0.18), '.kotlin'(-0.18), 'erule'(-0.17), 'kaar'(-0.17) |
| **21** | 'A'(0.15), ' singular'(0.14), ' A'(0.13), ' merely'(0.13), '4'(0.12), 'okens'(0.12), ' advisers'(0.12), 'ariant'(0.12), 'altung'(0.12), 'ilig'(0.11) | '.DropDown'(-0.23), 'iš'(-0.21), 'PELL'(-0.20), 'Ｅ'(-0.20), '/gtest'(-0.20), '.kotlin'(-0.19), 'etine'(-0.19), 'Ｔ'(-0.19), 'Ơ'(-0.19), ' scn'(-0.19) |
| **22** | 'okens'(0.15), 'ads'(0.15), 'olders'(0.15), 'A'(0.14), 'ilig'(0.14), 'DialogTitle'(0.14), 'altung'(0.14), 'ariant'(0.13), ' A'(0.13), 'SessionFactory'(0.13) | '.**DropDown**'(-0.27), 'Ｔ'(-0.21), 'Ｅ'(-0.20), '.kotlin'(-0.20), 'kaar'(-0.20), 'Ơ'(-0.20), 'ěř'(-0.20), 'Ｉ'(-0.20), '/gtest'(-0.20), '.unknown'(-0.19) |
| **23** | 'A'(0.21), 'ち'(0.18), 'मर'(0.17), ' A'(0.17), 'olders'(0.17), 'ilig'(0.16), 'ief'(0.16), 'ialized'(0.16), 'okens'(0.16), 'uxt'(0.16) | ' **myself**'(-0.22), 'iš'(-0.21), ' Eudicots'(-0.20), '我'(-0.19), ' **unsure**'(-0.19), 'isku'(-0.19), 'esiz'(-0.19), '**Ｅ**'(-0.18), ' saya'(-0.18), 'hiba'(-0.18) |
| **24** | 'A'(0.24), ' A'(0.21), ' B'(0.19), ''(0.18), 'C'(0.17), 'uxt'(0.17), 'ilig'(0.17), 'ief'(0.17), 'ち'(0.17), ' titul'(0.16) | ' Eudicots'(-0.23), ' **unsure**'(-0.23), ' **myself**'(-0.23), 'iš'(-0.22), ' Pazar'(-0.22), '我'(-0.21), '.Empty'(-0.21), 'hiba'(-0.20), 'eç'(-0.20), '歐'(-0.20) |
| **25** | 'A'(0.26), ' A'(0.25), ' B'(0.23), ' C'(0.23), ' D'(0.21), 'ち'(0.21), 'ief'(0.20), 'D'(0.20), 'ilig'(0.19), 'C'(0.19) | ' **E**'(-0.41), ' Eudicots'(-0.35), ' E'(-0.34), ' Е'(-0.30), 'E'(-0.29), ':E'(-0.29), '	E'(-0.29), 'Ｅ'(-0.29), '.E'(-0.28), 'Е'(-0.28) |
| **26** | 'A'(0.30), ' A'(0.26), 'C'(0.25), 'D'(0.25), ' C'(0.25), 'ち'(0.23), 'ief'(0.23), ' Rey'(0.22), '=A'(0.21), 'hasOne'(0.21) | ' E'(-0.45), ' Eudicots'(-0.37), ' E'(-0.35), ' Е'(-0.34), ':E'(-0.31), '	E'(-0.31), '.E'(-0.30), '_E'(-0.30), ' eoq'(-0.29), 'eç'(-0.29) |
| **27** | ' A'(0.39), 'A'(0.37), ' C'(0.31), 'C'(0.31), 'D'(0.27), 'ief'(0.26), ' A'(0.24), ' perme'(0.24), 'B'(0.24), ' B'(0.24) | 'eç'(-0.39), '//{{'(-0.36), 'еко'(-0.34), ' eoq'(-0.34), ' Eudicots'(-0.34), 'mada'(-0.34), ' ngang'(-0.34), '×\\n\\n'(-0.33), 'Ｅ'(-0.33), '.DropDown'(-0.33) |
| **28** | 'aed'(0.31), 'C'(0.30), 'A'(0.29), ' urlpatterns'(0.29), ' C'(0.29), ' Rey'(0.28), 'AGER'(0.28), 'ief'(0.27), ' A'(0.27), 'uxt'(0.27) | ' Eudicots'(-0.35), ' eoq'(-0.35), ' E'(-0.34), 'mî'(-0.34), 'mada'(-0.33), ' ngang'(-0.33), 'eç'(-0.32), 'erule'(-0.32), ' Е'(-0.31), 'Ｅ'(-0.31) |
| **29** | 'C'(0.42), 'A'(0.40), ' subt'(0.37), ' C'(0.35), ' A'(0.32), 'aed'(0.32), 'uxt'(0.32), 'B'(0.31), 'ief'(0.31), 'D'(0.31) | ' E'(-0.47), '_EOL'(-0.41), 'mada'(-0.40), ' eoq'(-0.40), ' Е'(-0.39), ' е'(-0.38), 'ọt'(-0.38), 'erule'(-0.38), '	E'(-0.37), 'PEND'(-0.37) |
| **30** | ' C'(0.43), ' A'(0.41), 'C'(0.41), 'A'(0.40), ' subt'(0.37), 'ា'(0.37), 'AGER'(0.36), 'ief'(0.35), ' A'(0.34), 'ange'(0.34) | ' E'(-0.80), ' E'(-0.63), 'E'(-0.61), '	E'(-0.61), ' Е'(-0.60), '_E'(-0.58), '.E'(-0.57), ':E'(-0.54), 'Е'(-0.53), 'Ｅ'(-0.53) |
| **31** | 'A'(0.59), 'C'(0.58), '.'(0.55), ' A'(0.53), 'B'(0.52), '…\\n'(0.52), ' C'(0.49), ' commodo'(0.44), ' B'(0.44), '.C'(0.44) | '**Ｉ**'(-0.80), '	E'(-0.70), 'Ｅ'(-0.70), ' E'(-0.69), 'ôi'(-0.69), 'amik'(-0.68), '""E'(-0.67), ' Ί'(-0.67), '//{{'(-0.66), '_EOL'(-0.66) |
| **32** | ' ACC'(1.62), 'ActivityIndicatorView'(1.59), ' Attention'(1.58), ' AC'(1.51), '_AC'(1.47), ' Explanation'(1.46), 'ocab'(1.45), ' Sabb'(1.43), 'APPING'(1.43), 'ACC'(1.41) | '*I'(-2.28), 'I'(-2.26), ' I'(-2.13), '(I'(-1.93), ' ""?'(-1.73), ' ""?""'(-1.73), '""I'(-1.71), ' **unknown**'(-1.71), '?:'(-1.67), '_I'(-1.67) |

---

### **L2 Norm**

![image.png](attachment:e60e7a7f-e8e3-4a5b-b15f-a062b4ef80da:image.png)

![image.png](attachment:67d7eb9a-9285-408a-b89a-cf09ee667b66:image.png)

  
 $\Delta h(l) = h_{\text{exp}}(l) - h_{\text{non}}(l)$
$\|\Delta h(l)\|_2 = \sqrt{\sum_i (\Delta h_i(l))^2}$

**1. 11–19 层 = 写入端 (Injection Zone)**

- 10 层以前几乎是 0
- 11 层开始 Δh 明显上升
- 19 层时已经是一个稳定、连续的上升斜率

这是经典的：Role signal is injected into the residual stream starting around L11.

这与你 activation patching 发现的 “中间层负责写角色/信心” 完全一致。

**注：写入（injection）不需要产生“大”Δh，也不需要维持方向** 它只需要：加入一个 consistent 差异，让 decoder 的后层能够 amplify，因此 injection zone 不会像 highway 一样爆炸式增长。

**2. 20–31 层 = Highway（Amplification Zone）**

从 layer 20 开始：

- Δh 的幅值加速上升
- 增长速度几乎呈递增趋势
- 和 Pearson correlation 的高层 band 完全重叠

这说明：模型把在 11–19 写入的 role direction，从 20 层开始持续放大与维持，形成一条“自信/专家方向”的高速通道。

**3. 32–33 层 = Logit Readout 放大器 (Final Writers)**

最后两层（特别是 33）出现爆炸式 L2 上升：

- 这是 logit readout 层
- residual stream 最终作用最大
- Δh 最终直接被映射到词表

这是非常典型的：Highway → Readout 的末端放大。

**完整逻辑链：**

**(A) Activation patching：11–19 层 causal**

→ 改这几层，模型角色判别 & reasoning willingness & confidence 全部翻转。

→ 所以 **injection 层在 11–19**。

**(B) Pearson correlation：20–32 层形成 highway band**

→ 显示 role signal 在高层被 maintained across layers。

→ 所以 **highway 在 20–32**。

**(C) Δh L2 Norm：11–19 上升、20–32 放大、33 爆发**

→ 完美解释 injection → amplification → readout 三阶段结构。

---

### **Transformer Dynamics（2025）**

**⭐ 1.支持：Residual Stream 的维持性（persistence）与高跨层相关性**

你的 repeated index（垂直条纹）假说提出：某些 neuron index 可能作为“信号通道”，在多个层中持续写入同一个方向。

但 logits lens 回馈：**单 neuron 对 unembedding 没意义 → 垂直 index 不可解释。**

**这篇 paper 的结论：**Residual Stream units exhibit consistently high correlations across layers, despite RS being a non-privileged . basis（Fig. 1C–1D）
意思是：**单个维度在层间高度相似，并不是因为它代表某个语义，只是因为 RS 的结构让它自然保持连续。**

这是一个直接的支持：Repeated-index 不是真正的“特定内容通道”，**而是 Residual Stream 的动力结构让维度自然而然跨层一致。**

这解释了你看到的 vertical stripe，但它不具备语义解释力。

**⭐ 2.支持：Your Δh L2-norm “growing magnitude in late layers” 是 paper 的关键发现之一**

你的 Δh magnitude 图显示：

- 层 1–10 几乎没有差异（role effect 很弱）
- 11–19 有增长（你们 paper 用的层区间正好在这个范围）
- 20 之后强烈“放大”（role signal amplification）

而这篇 paper 得到：RS vectors accelerate in later layers; velocity increases sharply in later layers（Fig. 1F）

也就是说：**later layers 会把任意微小差异（例如专家 vs 非专家）放大成巨大 Δh。**

这完美吻合你现在看到的 dense-RSN Δh L2 norm：

- 早期层没有差异（系统还没 amplify）
- 中层 11–19 层是 transition zone（维持性高，信息开始转化）
- 后期层放大成巨大的 Δh（模型为了最终 logit 调整 role-specific 信号）

**⭐ 3. 支持：为什么 Dense Δh 能解释，而 Sparse index-level 完全不能解释？**

你的结果显示：

- dense Δh（原始 hidden state difference）非常有解释力
- sparse 单 neuron logits lens 完全无法解释 role 信号

这篇 paper 给出了核心原因：RS is not a privileged basis 信息不是由 neuron 基底来表示，而是 distributed / rotational

并且：Individual units trace unstable periodic orbits 单 neuron activation 没有固定语义，是旋转的相空间动力（Fig. 2）

这意味着：**单个 neuron 的值没有稳定的功能解释** → 所以你的 logits lens neuron-level 当然看不到语义

**✔ 功能信号来自 RS 中的“方向”，不是“神经元”** → dense Δh 正是方向 → 解释力自然比 sparse neuron 强

**4.支持：为什么 repeated-index 没有语义？**你以为 vertical stripe（repeated index）反应“同一个 neuron 在传播 role”，

但现在发现：

- neuron-level 不可解释（logits lens 失败）
- dense-level 可解释（Δh 看得出 expert-suppression）
- 你的 repeated index 和 PCA-based direction overlap 非常低（0.03～0.19）

这篇 paper给出原因：Even though RS units remain highly correlated across layers, MI sharply decreases early 表示 non-linear components are transformed away.（Fig. 1H）

意思是：**neuron-level correlation 只是“动力连续性”，不是语义连续性**

→ 所以 repeated index 仅仅因为 RS 的结构，而不是因为它编码 role

这直接解释你的 repeated neuron phenomenon：

- “看起来重复”
- “但没有意义”

你现在的实验完全验证了 paper 的预测。

**⭐ 5.支持你对 Layer 11–19 的强调**

你的 paper 论述中，中层（11–19）是 role-effect 的关键窗口。

**这篇 paper 的结果：**

- 前 10 层 mutual information 急速下降 → 完成“非线性表示折叠”
- 11–19 层：representation 开始沿低维轨迹“straighten out”（Fig. 3B）
- 20 层后：amplification zone（velocity 急升）

**所以 11–19 层正是模型“稳定 channel”形成的区间。**

→ 这完全支持你们的 activation patching 选择在 11–19 层有效。

→ dense-RSN 在 11–19 很稳定，但在 20+ 层被大幅 amplify。

这是非常完美的对齐。

**⭐ 6.支持你提出的 “Highway Hypothesis”**

你的 Highway Hypothesis = 模型在 RS 中构建一个专用方向、并在层间维持/放大它。

这篇 paper 在 3 个地方完全支持你：

**✔ 高层维持性（unit-wise correlation high）**(Fig. 1C)

**✔ residual vectors follow low-dimensional trajectories** (Fig. 3 & 4)

**✔ trajectories straighten & amplify in late layers** (Fig. 4A)

所以你的 Highway Hypothesis 不是牵强，它是现在 transformer dynamical systems 的主流观点。

| **你的现象** | **paper 的解释** | **支持力度** |
| --- | --- | --- |
| dense Δh 在后期 layer 爆发 | RS velocity late-layer acceleration | ⭐⭐⭐⭐⭐ |
| 11–19 层是关键窗口 | MI, dimensionality collapse, trajectory straighten | ⭐⭐⭐⭐⭐ |
| repeated index 不具解释力 | RS basis is non-privileged; units rotate | ⭐⭐⭐⭐⭐ |
| sparse logits lens neuron-level 全部失败 | neuron-level 不稳定，不可语义化 | ⭐⭐⭐⭐⭐ |
| dense RSN 非常可解释 | signals exist as directions | ⭐⭐⭐⭐⭐ |
| highway / channel hypothesis | RS dynamics = stable low-d trajectories | ⭐⭐⭐⭐⭐ |

$\text{Mediation Ratio} = \frac{(Role \to RSN) \times (RSN \to Confidence)}{\text{Total Effect of Role on Confidence}}$