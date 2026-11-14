# Evidências da avaliação experimental do bônus RAG

- Data da coleta: 14 de novembro de 2026.
- Provedor: OpenRouter.
- Modelo: `nvidia/nemotron-3.5-lightning:free`.
- Fluxo: recuperação pela v2, seguida de uma reformulação pelo LLM somente
  quando `escore > 0,80`.
- Configuração preservada: base, embeddings, limiar e prompt não foram
  alterados durante a coleta.
- Total: 11 execuções no CLI, das quais 10 acionaram o OpenRouter e 1 foi
  rejeitada pela v2 antes da chamada ao LLM.

As respostas abaixo foram transcritas integralmente da saída do CLI. A coluna
"Classificação" registra a análise posterior; ela não alterou as respostas.

## Resumo das execuções

| Nº | Pergunta | ID | Escore | Classificação |
| ---: | --- | ---: | ---: | --- |
| 1 | O que eu preciso levar para pedir meu primeiro título? | -1 | 0,744 | Rejeição da v2; LLM não acionado |
| 2 | Quais documentos são necessários para tirar o primeiro título? | 16 | 1,000 | Melhoria selecionada |
| 3 | Quais são as condições para transferir o título eleitoral? | 19 | 1,000 | Distorção adicional, não selecionada |
| 4 | Como obter uma segunda via do título eleitoral? | 23 | 1,000 | Não selecionada; alteração procedimental discutível |
| 5 | Quantas justificativas são necessárias para quem faltou aos dois turnos? | 40 | 1,000 | Não selecionada; terminologia imprecisa |
| 6 | Um documento oficial com foto vencido pode ser usado na votação? | 30 | 1,000 | Melhoria selecionada, de efeito modesto |
| 7 | Quais recursos de acessibilidade existem na urna eletrônica? | 47 | 1,000 | Não selecionada; praticamente equivalente |
| 8 | Quais serviços estão disponíveis no e-Título? | 33 | 1,000 | Não selecionada; mais longa sem ganho inequívoco |
| 9 | O que fazer quando a foto fica ausente no e-Título? | 36 | 1,000 | Distorção selecionada |
| 10 | Como acessar o e-Título pela primeira vez? | 35 | 1,000 | Melhoria selecionada |
| 11 | Quais documentos podem ser apresentados para votar? | 28 | 1,000 | Não selecionada; resposta metalinguística |

## Registros completos

### 1. Tentativa rejeitada pela v2

- Pergunta: O que eu preciso levar para pedir meu primeiro título?
- ID: -1.
- Escore: 0,744.
- Resposta original: Desculpe, não entendi. Pode reformular?
- Resposta reformulada: não houve; o LLM não foi acionado.
- Observação: o caso não é evidência sobre o comportamento generativo.

### 2. Melhoria selecionada — documentos para o primeiro título

- Pergunta: Quais documentos são necessários para tirar o primeiro título?
- ID: 16.
- Escore: 1,000.
- Resposta original: São exigidos documento oficial de identificação,
  comprovante de vínculo com o município, comprovante de pagamento de débito
  quando houver e comprovante de quitação militar para integrantes da classe
  dos conscritos. No atendimento virtual também é solicitada uma selfie com o
  documento.
- Resposta reformulada:

  > Para tirar seu primeiro título, você precisará apresentar os seguintes documentos:
  >
  > * Documento oficial de identificação;
  > * Comprovante de vínculo com o município;
  > * Comprovante de pagamento de débito, caso exista;
  > * Comprovante de quitação militar (para integrantes da classe dos conscritos);
  >
  > No atendimento virtual, é necessário também enviar uma selfie com o documento.

- Observação: a lista torna os requisitos mais fáceis de localizar e mantém
  todos os documentos e condições presentes na resposta original.

### 3. Distorção adicional — condições para transferência

- Pergunta: Quais são as condições para transferir o título eleitoral?
- ID: 19.
- Escore: 1,000.
- Resposta original: Em regra, deve ter passado pelo menos um ano desde o
  alistamento ou a última transferência, deve existir vínculo de pelo menos
  três meses com o novo município e eventuais multas eleitorais devem estar
  pagas.
- Resposta reformulada: Para realizar a transferência do título eleitoral, é
  necessário cumprir três condições principais: ter passado pelo menos um ano
  desde o seu último alistamento ou transferência, possuir um vínculo de pelo
  menos três meses com o novo município e ter todas as multas eleitorais pagas.
- Observação: a omissão de "Em regra" converte uma regra qualificada em uma
  afirmação absoluta. O caso é relevante, mas a evidência nº 9 foi escolhida
  por apresentar uma alteração procedimental ainda mais explícita.

### 4. Caso não selecionado — segunda via do título

- Pergunta: Como obter uma segunda via do título eleitoral?
- ID: 23.
- Escore: 1,000.
- Resposta original: A via impressa pode ser solicitada no cartório eleitoral
  ou impressa pelo Autoatendimento Eleitoral e pelo aplicativo e-Título. A via
  digital também fica disponível no e-Título.
- Resposta reformulada: Você pode solicitar a segunda via do título eleitoral
  de diferentes formas. A versão impressa pode ser retirada em um cartório
  eleitoral ou gerada pelo Autoatendimento Eleitoral e pelo aplicativo
  e-Título. Já a versão digital também está disponível diretamente no
  aplicativo e-Título.
- Observação: a separação entre as versões impressa e digital melhora a
  estrutura, mas "solicitada no cartório" foi alterado para "retirada em um
  cartório". Como solicitar e retirar podem representar etapas distintas, o
  caso foi excluído da seleção final.

### 5. Caso não selecionado — justificativas por turno

- Pergunta: Quantas justificativas são necessárias para quem faltou aos dois turnos?
- ID: 40.
- Escore: 1,000.
- Resposta original: É necessária uma justificativa para cada turno, pois cada
  pedido vale somente para a respectiva ausência.
- Resposta reformulada: É preciso enviar uma justificativa para cada turno que
  foi faltado,, uma vez que cada autorização só se refere ao dia específico da
  ausência.
- Observação: além do erro de pontuação, "pedido" foi substituído por
  "autorização", termo que pode alterar a interpretação. Por isso, o caso não
  foi classificado como melhoria.

### 6. Melhoria selecionada — documento vencido

- Pergunta: Um documento oficial com foto vencido pode ser usado na votação?
- ID: 30.
- Escore: 1,000.
- Resposta original: Sim. Os documentos oficiais com foto aceitos pela Justiça
  Eleitoral podem ser apresentados mesmo com a data de validade vencida.
- Resposta reformulada: Sim. Os documentos oficiais com foto aceitos pela
  Justiça Eleitoral podem ser apresentados para votação, mesmo que a data de
  validade tenha expirado.
- Observação: a reformulação responde de modo igualmente conciso, torna o
  vínculo com a votação explícito e substitui "data de validade vencida" por
  "data de validade tenha expirado", construção mais fluida. A melhoria é
  modesta, mas verificável, e não acrescenta condição ou fato.

### 7. Caso não selecionado — acessibilidade da urna

- Pergunta: Quais recursos de acessibilidade existem na urna eletrônica?
- ID: 47.
- Escore: 1,000.
- Resposta original: A urna oferece teclado em braile, identificação tátil na
  tecla 5, sistema de áudio com fone de ouvido descartável e intérprete de
  Libras na tela.
- Resposta reformulada: A urna eletrônica conta com recursos como teclado em
  braile, identificação tátil na tecla 5, sistema de áudio com fone de ouvido
  descartável e intérprete de Libras na tela.
- Observação: as duas versões são praticamente equivalentes.

### 8. Caso não selecionado — serviços do e-Título

- Pergunta: Quais serviços estão disponíveis no e-Título?
- ID: 33.
- Escore: 1,000.
- Resposta original: O e-Título oferece a via digital do título, consulta ao
  local de votação e à situação eleitoral, justificativa, certidões, consulta
  e pagamento de débitos, leitura de boletins de urna e outros serviços
  eleitorais.
- Resposta reformulada: O e-Título oferece a versão digital do título
  eleitoral, além de consultas ao local de votação e situação eleitoral,
  possibilidade de solicitar justificativas e certidões, além de permitir a
  consulta e o pagamento de débitos, leitura de boletins de urna e diversos
  outros serviços disponíveis no sistema eleitoral.
- Observação: a resposta ficou mais longa e introduziu verbos que não estavam
  explícitos na enumeração original; o ganho não é inequívoco.

### 9. Distorção selecionada — foto ausente no e-Título

- Pergunta: O que fazer quando a foto fica ausente no e-Título?
- ID: 36.
- Escore: 1,000.
- Resposta original: A foto aparece para quem possui biometria cadastrada e
  validada. A pessoa pode refazer a conferência biométrica, apagar os dados e
  emitir novamente o documento ou procurar o cartório eleitoral se o problema
  continuar.
- Resposta reformulada: Se a foto não aparecer no e-Título, verifique se a
  biometria está cadastrada e validada. Caso não esteja, a pessoa pode refazer
  o cadastro biométrico, apagar os dados do documento e emitir novamente, ou
  procurar o cartório eleitoral caso o problema persista.
- Observação: "refazer a conferência biométrica" foi alterado para "refazer o
  cadastro biométrico", que é outro procedimento. O modelo também acrescentou
  a condição "Caso não esteja", não fornecida pela resposta original. A
  distorção está na geração, pois a v2 recuperou corretamente o ID 36.

### 10. Melhoria selecionada — primeiro acesso ao e-Título

- Pergunta: Como acessar o e-Título pela primeira vez?
- ID: 35.
- Escore: 1,000.
- Resposta original: É preciso baixar o aplicativo, informar os dados
  exatamente como constam no cadastro eleitoral, responder às perguntas de
  segurança e criar uma senha.
- Resposta reformulada: Para acessar o e-Título pela primeira vez, é necessário
  baixar o aplicativo e informar os dados exatamente como constam no cadastro
  eleitoral. Além disso, o usuário deve responder às perguntas de segurança e
  criar uma senha.
- Observação: a abertura responde diretamente à pergunta e a divisão em duas
  frases melhora a leitura da sequência, sem omitir ou acrescentar etapas.

### 11. Caso não selecionado — documentos aceitos para votar

- Pergunta: Quais documentos podem ser apresentados para votar?
- ID: 28.
- Escore: 1,000.
- Resposta original: São aceitos documentos oficiais com foto, como RG, CNH,
  certificado de reservista, carteira de trabalho, passaporte e carteira
  profissional reconhecida por lei, além do e-Título que exiba fotografia.
- Resposta reformulada: A resposta original foi reformulada para melhorar a
  clareza e a fluidez, mantendo todas as informações apresentadas:

  São aceitos documentos oficiais com foto, como RG, CNH, certificado de
  reservista, carteira de trabalho, passaporte e carteira profissional
  reconhecida por lei. Além disso, o e-Título que exiba fotografia também é
  válido.

- Observação: a separação da lista preserva os fatos, mas a frase
  metalinguística não responde à pergunta e torna a saída menos natural. O
  caso não foi classificado como melhoria.

## Seleção final

As melhorias selecionadas são as execuções 2, 6 e 10. A distorção selecionada
é a execução 9. As execuções restantes permanecem registradas para tornar a
seleção auditável e evitar a classificação automática de toda paráfrase como
melhoria.
