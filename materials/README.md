# Materials

機械学習を学ぶための教材

## Machine Learning

* [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning)
  * 最初はここから。ほかのどんな書籍やブログよりよい。
* [fast.ai](http://www.fast.ai/)
  * プログラマーのための機械学習講座といったコース。実践が多く、理論だけでなく手を動かして学びたいという方にお勧め。
* [CSC 321 Winter 2018 Intro to Neural Networks and Machine　Learning](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/)
  * トロント大学のニューラルネット講座。DNNの実装に使用されるフレームワークを意識した解説になっていて、理論的な内容と実装のギャップが少なくなるよう工夫されている。PyTorchのチュートリアルもあり。
* [CS 188 | Introduction to Artificial Intelligence](https://inst.eecs.berkeley.edu/~cs188/fa18/)
  * Berkeleyで行われているAIに関する講義の資料。AIを合理的な行動=期待値を最大にする行動をとるエージェントとし、アルゴリズムによる最大化(CSP: 制約充足問題etc)、データによる知見からの最大化(強化学習、ベイジアン)、と展開している。
  * 「AIとは」からきちんとかみ砕いて解説してくれている
* [CS W182 / 282A](https://cs182sp21.github.io/)
  * バークレー大学の深層強化学習のコースCS W182が公開。講義資料だけでなく、動画も公開されている。
* [深層学習 (機械学習プロフェッショナルシリーズ)](https://www.amazon.co.jp/dp/4061529021)
  * ディープラーニングについて良くまとまった書籍
* [Python機械学習プログラミング](https://www.amazon.co.jp/dp/4844380605)
  * scikit-learnを雰囲気で使っていると思ったら参照するとよい書籍。
  * 実践編として、各種アルゴリズムをスクラッチから作る[ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch)もおすすめ。
* [ゼロから作るDeep Learning ――Pythonで学ぶディープラーニングの理論と実装](https://www.oreilly.co.jp/books/9784873117584/)
  * ニューラルネットワークを基礎的な部分から自分で作成していくことで動作について理解できる書籍。ただ、機械学習全般を扱っているわけではないのでその点は注意。
* [How to build a Recurrent Neural Network in TensorFlow](https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767)
  * ゼロから作るRNNといった趣の記事。TensorFlowで素の状態からRNNを作るところから、実際にRNNのモジュールを使って構築するところまでを完全にカバーしている。 
* [DEEP LEARNING NYU CENTER FOR DATA SCIENCE](https://atcold.github.io/pytorch-Deep-Learning/)
  * PyTorchを使用したDNNの講座。Energy-BasedやGraph Convolutionについて解説があるのは珍しいと思う。GitHubでJupyterも公開されており、日本語翻訳も提供されている。

### Additional

* [Stanford Artificial Intelligence Laboratory](http://ai.stanford.edu/courses/)
  * Stanfordの、AI関連講座のリンク集。基本はずさない。
* [NIPS HIGHLIGHTS, LEARN HOW TO CODE A PAPER WITH STATE OF THE ART FRAMEWORKS](https://mltrain.cc/events/nips-highlights-learn-how-to-code-a-paper-with-state-of-the-art-frameworks/)
  * NIPS2017で開催された、最新の研究をどうTensorFlowやPyTorchといった機械学習フレームワークのコードに落とし込んでいくのかというワークショップの資料
* [Seedbank](http://tools.google.com/seedbank/)
  * インタラクティブに動かせる機械学習のコードを集めたサイト。事前学習済みのGANモデルの使い方やPerformance RNNによる音楽生成といった応用例から、ニューラルネットワークの仕組みといった基礎的な内容まで幅広い。pandasの使い方などもある。
* [Resources for CS 229 - Machine Learning](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-supervised-learning.html)
  * 機械学習のポイントをまとめたチート集。あれなんだったかな、という時さっと見るのにとても良い。
* [Depth First Learning](http://www.depthfirstlearning.com/)
  * ある一つの論文について、その論文中に登場する基本的な概念などを掘り下げて解説するというスタイルの記事。
  * 機械学習の論文を読みたい、でも論文で前提知識とされている概念がわからない、普通の教科書を買って勉強して出直すしかないのか・・・そう感じている方にとってはベストな記事。
* [Fairness-Aware Data Mining](http://www.kamishima.net/archive/fadm.pdf)
  * 機械学習を扱うなら知っておきたい、モデルに内在するバイアス(性別や人種といった特徴量による判断の偏りなど)についての解説資料。
  * バイアスを生む要因、その避け方などがまとめられている。
* [Imperial College Mathematics department Deep Learning course](https://github.com/pukkapies/dl-imperial-maths)
  * PyTorchで深層学習を学ぶコース(TensorFlowもあるよう)。CNN、RNN、強化学習と扱っている話題が広く、コードも比較的きれいに書かれている。短いため、PyTorchがどんな感じかさくっと学びたい場合にお勧め。
* [第1回 ディープラーニング分散学習ハッカソン](https://www.cc.u-tokyo.ac.jp/events/lectures/111/?linkId=100000004406701)
  * 東京大学情報基盤センターが主催する、マルチGPUを使った学習のハンズオン。
  * 資料が公開されており、分散学習の仕組みと課題、そしてChainerMN/TensorFlowで実際に分散学習を行う方法までが解説されている。
* [AI-Sys Spring 2019](https://ucbrise.github.io/cs294-ai-sys-sp19/)
  * UC Berkeleyの、DNNを効率的に実行するための手法についての講義。実行の工夫として分散環境やネットワーク構造のコンパイル、モデル側の工夫として構造探索や蒸留(主に枝刈り)の話題が取り上げられている。
* [Parameter optimization in neural networks](https://www.deeplearning.ai/ai-notes/optimization/)
  * Neural Netの最適化に関する解説記事。数式の説明・理解を助けるインタラクティブなデモ・手早く引けるOptimizer選択のチートシート、と至れり尽くせりな内容になっている。
* [fastai/fastbook](https://github.com/fastai/fastbook)
  * FastAIの講座が教科書に。Jupyter形式で書かれており、読みつつすぐに実践できる。
* [A Survey of Deep Learning for Scientific Discover](https://arxiv.org/abs/2003.11755)
  * DNNの研究・活用全体をガイドした資料。基本的な仕組み・タスク、効率的な学習方法・解釈性など近年の話題も含めて幅広いトピックが整理されて書かれている。実装に使えるOSSの紹介も行われている。
* [機械学習の研究者を目指す人へ](https://takahashihiroshi.github.io/contents/for_ml_beginners.html)
  * 機械学習の研究者を目指すために必要な知識や学習教材をまとめた一覧。

### Not Recommend :droplet:

* [深層学習](https://www.amazon.co.jp/dp/4048930621/)
  * 原著として扱っている内容が若干古く、また冗長なので読みにくい部分がある。機械学習の入門書やオンラインコースは近年かなり洗練されているので、あえて本書を最初に読む必要はないと思う。

## Mathematics

* [Numerical Linear Algebra for Coders](https://github.com/fastai/numerical-linear-algebra/blob/master/README.md) 
  * 用例を通じて学ぶ線形代数。Numpy, scikit-learn, PyTorchを利用した実装を通じてその背後の仕組みを学ぶというスタイル。
* [Introduction to Applied Linear Algebra – Vectors, Matrices, and Least Squares](http://web.stanford.edu/~boyd/vmls/)
  * ケンブリッジで使用されている線形代数の教科書。実際にどんなところで利用されているのか、という解説までついていてわかりやすい。
* [Reducing Dimensionality from Dimensionality Reduction Techniques](https://medium.com/towards-data-science/reducing-dimensionality-from-dimensionality-reduction-techniques-f658aec24dfe)
  * よく利用される、ただ誤用されがちな次元削除のテクニック(PCA/t-SNE/Auto Encoder)について、ゼロからの実装を交えて解説してくれている。
* [Python Numpy Tutorial](http://cs231n.github.io/python-numpy-tutorial/)
  * 画像認識で有名なCS231nにおける、Numpyのチュートリアル資料。線形代数的な解説はないが、Numpyの基本的な使い方が把握できる
* [Mathematics for Machine Learning](https://mml-book.github.io/)
  * 機械学習のための数学を学ぶ本。なんと無料で公開されている。基礎となる数学をきっちり学びたい人、実務で必要な箇所を学びたい人、それぞれが読者として想定されており、基礎パートと実応用パートがコンポーネント化されて書かれている。
* [The Modern Mathematics of Deep Learning](https://arxiv.org/abs/2105.04026)
  * DeepLeraningの成功を数学的に解析する研究をまとめたサーベイ。汎化する理由、深さの役割、次元の呪いの克服といったテーマについて、最新の研究がまとめられている。

## Vision

* [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
  * Stanfordの、Computer Visionの講座。これが鉄板。
  * こちらのコースを受講後に、まとめた記事。[解説編](http://qiita.com/icoxfog417/items/5fd55fad152231d706c2)と[実装編](http://qiita.com/icoxfog417/items/5aa1b3f87bb294f84bac)。
* [cvpr2017_gan_tutorial](https://github.com/mingyuliutw/cvpr2017_gan_tutorial)
  * CVPR2017でのGANのチュートリアル資料。GAN以前の手法の問題点から紹介しており、GAN登場の背景から仕組み、学習方法までを習得できる完全盤的資料。
* [A 2017 Guide to Semantic Segmentation with Deep Learning](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review)
  * 画像内の領域認識を行うセグメンテーションについて、その問題設定と現在取り組まれている手法の網羅的な紹介。
  * [POSTDで日本語化された](https://postd.cc/semantic-segmentation-deep-learning-review/)
* [Deep Learning for Videos: A 2018 Guide to Action Recognition](http://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review)
  * 上記のブログと同じ筆者(団体?)の手による、動画中の動作認識に関するサーベイ。
* [UNDERSTANDING DEEP LEARNING FOR OBJECT DETECTION](http://zoey4ai.com/2018/05/12/deep-learning-object-detection/)
  * 物体検出についての解説記事。Two-stage/Single-stageにきちんと分けて仕組みの解説が行われているほか、速度/精度のトレードオフや今後の研究課題まできちんとまとめられている。
* [Deep Learning for Generic Object Detection: A Survey](https://arxiv.org/abs/1809.02165)
  * 物体検知の手法のまとめで、手法だけでなく物体検知というタスク自体についてもその歴史をたどり系統図にまとめている。そのため単純に近年の手法だけでなく、体系的な系譜を学ぶことができる。
  * 図解も豊富で分かりやすく、まさに"Survey"の名を冠するにふさわしい出来。
* [画像認識と深層学習](https://www.slideshare.net/ren4yu/ss-234439652)
  * 画像認識モデルの進化がまとめられた資料。モジュール開発競争の過程と果てのAutoML、また枝刈り/蒸留等による軽量化など幅広い領域の歴史が網羅されている(物体検出もついている)。

### Additional

* [shayneobrien/generative-models](https://github.com/shayneobrien/generative-models)
  * VAE、また様々な/GANといった生成系モデルの解説実装を提供しているリポジトリ。コードがとても整理されていて、コードを読みながら学べる形。
* [deep learning object detection](https://github.com/hoya012/deep_learning_object_detection)
  * 物体検知に関する手法をまとめてくれているサイト。SOTAの更新履歴を追うことができる。論文のリンクと、公開されて入れば実装コードへのリンクも貼られている。
* [敵対的生成ネットワーク（GAN）](https://www.slideshare.net/cvpaperchallenge/gan-133159239)
  * GAN研究の体系的な整理、評価、応用、実装、研究を牽引している組織までがまとめられた資料。とても貴重。
* [モダンな深層距離学習 (deep metric learning) 手法: SphereFace, CosFace, ArcFace](https://qiita.com/yu4u/items/078054dfb5592cbb80cc)
  * 分類ではなく、ベクトル表現間の距離を学習するMetric Learningの手法についての記事(距離学習は、顔認証など分類数が不定/事前のサンプルが難しいケースで有効)。
  * 距離を直接学習する手法が主流だったが、最近は分類問題を解きつつ学習できるようになってきたよう。既存の距離学習について解説された記事へのリンクもある。
  * 基本的な手法であるTriplet Lossについてはこちらが詳しい: [Deep Metric Learning の定番⁈ Triplet Lossを徹底解説](https://qiita.com/tancoro/items/35d0925de74f21bfff14)
* [Deep Metric Learning: A Survey](https://www.researchgate.net/publication/335314481_Deep_Metric_Learning_A_Survey)
  * 深層学習によるMetric Learningのサーベイ。Metric Learningの重要な要素として学習データのサンプリング、ネットワーク構成、lossの3点を挙げており、特にサンプリングは(注目されがちな)他2つと同等に重要であるとしている。
* [画像キャプションの自動生成](https://www.slideshare.net/YoshitakaUshiku/ss-57148161)
  * 画像キャプションの自動生成について、歴史や研究、評価手法などについてまとめられた資料。
* [三次元点群を取り扱うニューラルネットワークのサーベイ Ver. 2 / Point Cloud Deep Learning Survey Ver. 2](https://speakerdeck.com/nnchiba/point-cloud-deep-learning-survey-ver-2)
  * 点群(Point Cloud)を扱う研究をまとめたサーベイ。基礎からカテゴリ別に膨大な量の論文が網羅されており、応用についてもまとめられている。
* [第126回 RSJロボット工学セミナー「 Visual SLAMと深層学習を用いた３Dモデリング」](https://www.slideshare.net/KenSakurada/126-rsj2020522)
  * 3Dモデリング、さらに時間軸を加えた4Dモデリングの研究動向、基本的な手法(SfM/SLAM・V-SLAM)の解説がまとめられた資料。実装に使えるライブラリまで紹介されている。
* [2020.09.30 NL研 招待講演 Vision&Languageの研究動向](https://speakerdeck.com/sei88888/2020-dot-09-dot-30-nlyan-zhao-dai-jiang-yan-vision-and-languagefalseyan-jiu-dong-xiang)
  * Vision&Languageの研究動向に関するまとめ。最新の研究だけでなく、1970~1980年代の萌芽的な研究から紹介されており、歴史的な流れが概観できる資料となっている。
* [Deep Learningを用いた経路予測の研究動向](https://speakerdeck.com/himidev/deep-learningwoyong-itajing-lu-yu-ce-falseyan-jiu-dong-xiang)
  * DNNを用いた経路予測の手法のサーベイ。衝突(インタラクション)を避けることができる経路を予測するタスクで、空間範囲の情報をまとめるPoolingや、この分野でもAttentionが用いられている。
* [Visual SLAM入門 ～発展の歴史と基礎の習得～](http://cvim.ipsj.or.jp/index.php?id=tutorial)
  * ロボットやARで空間を認識するために使用されるVisual SLAMを基礎から解説したチュートリアル資料。定義から代表的な3つの手法（特徴点、直接法、DNN）、3次元の推定まで幅広く解説されている。また、近年課題となっているプライバシーの問題にも言及されている。

### Not Recommend :droplet:

* [はじめてのパターン認識](https://www.amazon.co.jp/dp/4627849710)
  * 初心者にはあまり優しくない(輪講を途中でやめたことがある)。
* [パターン認識と機械学習](https://www.amazon.co.jp/dp/4621061224)
  * 同様に、初心者が最初のページから読んでいくには厳しい(分量の多さも挫折の要因となる)。
  * 読みたい方は、サポート資料として東大松尾研の方々がまとめた輪講用資料を参照するとよい: [PRML輪読資料](http://deeplearning.jp/seminar-2/#1515565927378-9a11fdc4-798a)
  * Numpyで紹介されているアルゴリズムを実装したリポジトリがある: [ctgk/PRML](https://github.com/ctgk/PRML)

なお、上記2つの書籍を機械学習「入門」図書として挙げている記事は基本的に参照する価値はないと断言できる。

* [実践 コンピュータビジョン](https://www.oreilly.co.jp/books/9784873116075/)
  * 画像処理初心者には優しくない書籍。


## NLP

* [CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/)
  * Stanfordの、自然言語処理におけるDNNの講座。これが鉄板。
* [Natural Language Processing](https://web.stanford.edu/~jurafsky/NLPCourseraSlides.html)
  * Stanfordの、DNNでない頃のNLPのコース資料。基本的なテキストの処理方法や言語モデルといったベーシックなところを扱ってくれている。
* [oxford-cs-deepnlp-2017/lectures](https://github.com/oxford-cs-deepnlp-2017/lectures)
  * Oxford大学で行われた自然言語の講義資料。言語モデルや音声認識、QAタスクなど幅広く扱い、それらでDNNをどう利用するかも述べられている。
  * [講義動画](https://www.youtube.com/playlist?list=PL613dYIGMXoZBtZhbyiBqb0QtgK6oJbpm)もある。
  * [簡単な日本語解説記事](http://englishforhackers.com/oxford-cs-deepnlp-2017-summary.html)
* [Natural Language Processing/Info 159/259. Fall 2018](http://people.ischool.berkeley.edu/~dbamman/nlp18.html)
  * UC Berkeleyで行われている自然言語処理のコース資料。自然言語処理の大分基本的なところから、ニューラル系の手法まで幅広く解説されている。
  * 資料内の事例も、テキスト分類ならTwitterやAmazonレビューなど興味を引くような題材で紹介されている。
* [[最新版] JSAI2018 チュートリアル「"深層学習時代の" ゼロから始める自然言語処理」](https://www.slideshare.net/yukiarase/jsai2018-101054060/yukiarase/jsai2018-101054060)
  * 人工知能学会2018でのチュートリアル資料。DNN以降の自然言語処理の基礎が入力データ(単語等)の「ベクトル化」であるとして、ではベクトル化の手法としてどんなものがあるのか、といった形で解説がされている。
  * 学習データが少ない時や前処理の必要性などにも触れられている。
* [NLP Course | For You](https://lena-voita.github.io/nlp_course.html)
  * Yandex School of Data Analysis (YSDA) で開講されている自然言語処理のコースの補足資料。(かわいい)図解が豊富で理解しやすい内容になっている。
* [Document Clustering with Python](http://brandonrose.org/clustering)
  * 自然言語の基本的な特徴量の抽出方法から、各クラスタリングのアルゴリズムを実際試してみるところまでを紹介している。
* [SAILORS 2017 NLP project](https://github.com/abisee/sailors2017)
  * Stanfordで行われている、女子高校生を対象として機械学習/自然言語処理を教えるプログラム[SAILOR](http://ai4all.stanford.edu/)で使用された教材。
  * 学習の手始めとしてもよいが、チュートリアル教材を作る際の参考にもなる。
* [CS 4650 and 7650](https://github.com/jacobeisenstein/gt-nlp-class)
  * ジョージア工科大学の自然言語処理の授業で使われる教科書。扱っているテーマは、完全版と思えるほど幅広い。
  * 図は少なめだが、シンプルな疑似コード的な記述が随所にあり分かりやすく解説されている。
* [A Review of the Neural History of Natural Language Processing](http://blog.aylien.com/a-review-of-the-recent-history-of-natural-language-processing/)
  * ニューラルネットで自然言語処理を扱う手法が発展してきた歴史を解説した記事。
  * 時系列の流れと、マイルストンとなった論文がまとまっており発展の流れを概観するのにとても良い。
  * スライドの資料がこちから参照できる: [Frontiers of Natural Language Processing](https://www.slideshare.net/SebastianRuder/frontiers-of-natural-language-processing)
* [全日本CV勉強会発表資料 Learning Transformer in 40 Minutes](https://speakerdeck.com/sei88888/quan-ri-ben-cvmian-qiang-hui-fa-biao-zi-liao-learning-transformer-in-40-minutes)
  * Transformerの速習ができるスライド。メリット・デメリットから基本ブロックの解説、学習形式までコンパクトにまとめられている。

## Classification

* [Reducing Toxicity in Language Models](https://lilianweng.github.io/lil-log/2021/03/21/reducing-toxicity-in-language-models.html)
  * 有害なコンテンツを検出/無害化する研究をまとめた記事。「有害」の定義、検出(単純には文書分類)、テキスト生成(Decode)時に言い方をマイルドにする無害化の研究が紹介されている(データセットも掲載されている)。
* [Best Research articles on Deep Learning for Text classification (2015–2016)](https://medium.com/towards-data-science/best-research-articles-on-deep-learning-for-text-classification-2015-2016-aaa7950af775)
  * テキスト分類におけるDNNの適用についてのサーベイ集。RNN/CNNを利用する際は目を通すとよい。
* [Deep Learning for Sentiment Analysis : A Survey](https://arxiv.org/abs/1801.07883)

### Information Retrieval

* [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)
  * 情報抽出に関する教科書。わかりやすさに定評のあるCristopher先生の本
* [サービス特性にあった検索システムの設計戦略](https://techlife.cookpad.com/entry/2019/11/18/110000)
  * 開発期間、メンテナンスコスト、また検索特性に応じた検索システムの構築を解説した記事。インフラとしてAWSを使うかどうか、DB(インデックス)を含めた構成はどうするか、検索に利用するテキスト、検索の評価指標は、など各ポイントについて検討内容がしっかりと書かれている。
* [検索体験を向上する Query Understanding とは](https://recruit-tech.co.jp/blog/2019/12/25/query-understanding-overview/)
  * Query Understanding(イメージ的には検索のUX)について解説された記事。検索者の意図に応じて(既知の情報を補完したいのか、未知の情報を見つけたいのか)、検索システムができること(検索者への働きかけ/検索システム側で意図を理解する方法)は何か、という点がまとめられている。
* [Building a Better Search Engine for Semantic Scholar](https://medium.com/ai2-blog/building-a-better-search-engine-for-semantic-scholar-ea23a0b661e7)
  * 検索を機械学習で改善する難しさを解説した記事。一般的にデータ(ログ)が多ければ多いほど精度が改善されるはずだが、1/3ほどは役に立たなかったとしている。例として検索したときすでに知っているページはクリックしないことがままあるが、こうしたユーザーの挙動は機械学習的にはノイズになるなど。


### Dialog

* [対話システム (自然言語処理シリーズ)](https://www.amazon.co.jp/dp/433902757X)
  * 対話研究の歴史、またその実装のアプローチまで、網羅的に書かれたまさに学習の起点となる本
  * こちらを[まとめた記事](http://qiita.com/Hironsan/items/6425787ccbee75dfae36)
* [Deep Learning for Dialogue Systems](https://www.csie.ntu.edu.tw/~yvchen/doc/DeepDialogue_Tutorial.pdf)
  * ACL 2017での、対話システムのチュートリアル資料。ベーシックなフレーム型の対話からDNNまで、網羅的に解説されている。
* [Open-Domain Neural Dialogue Systems](https://www.csie.ntu.edu.tw/~yvchen/doc/OpenDialogue_Tutorial_IJCNLP.pdf)
  * IJCNLP 2017での、オープンドメインな対話システムのチュートリアル資料。
* [Introduction to Visual Question Answering: Datasets, Approaches and Evaluation](https://tryolabs.com/blog/2018/03/01/introduction-to-visual-question-answering/)
  * Visual QAのタスクについて、データセット、ベースライン、評価手法、将来の応用など研究を始めるのに必要なことが一通り書かれた記事(データセットについては問題点まで指摘されている)。
* [End to-end goal-oriented question answering systems](https://www.slideshare.net/QiHe2/kdd-2018-tutorial-end-toend-goaloriented-question-answering-systems-110817295)
  * 自然言語でQAを行う手法について、未だかったないほどに手法がまとめられた資料。LinkedInでの取り組みについても書いてあるというおまけ付き。
* [Advancing the State of the Art in Open Domain Dialog Systems through the Alexa Prize](https://arxiv.org/abs/1812.10757)
  * 対話ボットの開発コンテストであるAmazon Alexa Prizeの中で生まれた開発ツールキット(CoBot)と、研究成果の紹介。対話システムにおける手法だけでなく、実践的なインフラ構成についてまで知ることができる良い資料。
* [What makes a good conversation?](https://ai.stanford.edu/blog/controllable-dialogue/)
  * 「良い」対話システムをどう作るかを解説した記事。
  * 一般的な質問(どれくらい発話内容をコントロールできるかなど)への端的な回答をまず示し、その背景技術を解説している。手法は学習データがある程度ある場合/ない場合に使える手法がそれぞれ紹介されており、実践的。

### Representation

* [Representations for Language: From Word Embeddings to Sentence Meanings](https://nlp.stanford.edu/~manning/talks/Simons-Institute-Manning-2017.pdf)
  * 自然言語の表現学習について、分散表現(word2vec/GloVe)を皮切りに現時点の鉄板であるBidirectional-LSTM + Attention、またCNNの適用などの手法について解説している。この資料だけで、現在の表現学習を概観できる。
  * [Video](https://simons.berkeley.edu/talks/christopher-manning-2017-3-27)
* [From Word to Sense Embeddings: A Survey on Vector Representations of Meaning](https://arxiv.org/abs/1805.04032)
  * 単語分散表現の場合一つの単語は一つのベクトルで表現されるが、本来単語は多様な意味を持つ。そうした「意味」の表現の獲得を目指した研究のサーベイ。教師なし/知識ベースの大きく2つに分けて解説されており、その比較についても記載されている。
* [Deep Learning for Semantic Composition](http://egrefen.com/docs/acl17tutorial.pdf)
  * 文の意味解釈を行うタスクにDNNを適用する際の手法について、体系的なまとめ。
* [Learning with Latent Linguistic Structure](http://www.phontron.com/slides/neubig18blackbox.pdf)
  * 自然言語における構造を学ぶ手法についての解説資料。
  * 論理推論やプログラムなどの複雑な構造をもつドメインではアノテーションを行うのが難しいため、潜在構造を推定しつつ半教師ありの手法を適用するStructVAEについて解説されている。
* [Disentanglement Survey:Can You Explain How Much Are Generative models Disentangled?](https://www.slideshare.net/HidekiTsunashima/disentanglement-surveycan-you-explain-how-much-are-generative-models-disentangled)
  * Disentanglementの研究動向についてまとめられた資料。手法、また評価についてどのように研究が進展していっているのか理解できる。
* [EMNLP2019-Spec-Tutorial](https://docs.google.com/presentation/d/1QwD6Vd-SWJJWdR-QmAHWYDlxfHHeKTmEznDdIZg5aag/edit#slide=id.g706653db9c_0_80)
  * 分散表現を文脈に合わせてチューニングする手法をまとめた資料。同じ文脈で出てくる単語は分散表現上距離が近くなるので、(価格の)高い/安い、地名などは意味が異なるものの距離は近くなってしまう。そこで、外部知識などを用いてチューニングしようという研究。
* [学習済み日本語word2vecとその評価について](https://blog.hoxo-m.com/entry/2020/02/20/090000)
  * 適当に使ってしまいがちな分散表現をしっかりと評価した記事。類似度とアナロジー2つのタスクについて、いずれも日本語の評価データセットを使って評価している。
  * 形態素が評価データセットのものと合わないなど細かいエラーを丁寧につぶしている。chiVeが良いがfastTextも手ごろか。
* [Embeddings in Natural Language Processing](http://josecamachocollados.com/book_embNLP_draft.pdf)
  * 分散表現特化の解説本。単語/文の表現、BERTに代表される文脈考慮の表現はもちろん、グラフベースの分散表現も解説されている。
  * 近年問題とされているバイアスの問題にも触れられており、かなりの良書。
* [Representation Learning and NLP](https://link.springer.com/chapter/10.1007/978-981-15-5573-2_1)
  * 自然言語処理の表現学習の本が公開。BERTのあたりまで扱ってくれている。OpenAccessなのでフリーで読むことが可能。

### Named Entity Recognition (NER)

* [A Survey on Recent Advances in Named Entity Recognition from Deep Learning models](http://www.aclweb.org/anthology/C18-1182)
  * 固有表現認識の手法に関するサーベイ。単に研究だけでなくデータセットや評価指標の解説もあり、Recentとついているが旧来の手法についても言及されていて既存のサーベイの紹介もしてくれているなど、短いながらよくまとめられた資料。

### Others

* [Survey of the State of the Art in Natural Language Generation: Core tasks, applications and evaluation](https://github.com/arXivTimes/arXivTimes/issues/563)
* [Machine Learning on Source Code](https://ml4code.github.io/)
  * 機械学習をシステム開発に役立てる研究のサーベイ。List of Papersには2007年のからの研究がずらりと並ぶ。コードの補完や訂正に関するものが多いが、コミットメッセージやコメントの生成、画面からのコード生成といった珍しいものもある。
* [トピックモデル (機械学習プロフェッショナルシリーズ)](https://www.amazon.co.jp/dp/4061529048)
  * トピックモデルについてはこちらの書籍がとても分かりやすい。
* [All About NLP (AAN)](http://tangra.cs.yale.edu/newaan/)
  * 自然言語処理に関連する論文やチュートリアルなどを集めたポータルサイト。Yale大学が運営しており、共同プロジェクトの実施やデータセットの公開なども行っている。
* [Algorithms - Lede 2018 @ Columbia Journalism School](https://github.com/jstray/lede-algorithms)
  * コロンビア大学ジャーナリズム大学院における、ジャーナリズムで使用されるアルゴリズムについての講義資料。
  * 前半は一般的な自然言語処理・機械学習の解説だが、後半はそれによりもたらされうるメディアバイアスやその原因となるデータ品質/予測結果の解釈方法などが解説されている。
  * 「機械学習の基本的な内容とその利用に関する注意」を学ぶにはよい資料。
* [Neural Reading Comprehension and Beyond](https://purl.stanford.edu/gd576xb1833)
  * 機械学習で文書読解をさせるMachine Comprehensionについての体系的なまとめ。
  * 紹介されている手法自体は多くないが、特徴ベースの手法からニューラルまで解説されており、また周辺の議論(QAとの違いや、評価方法など)についてもきちんとまとめられている。
* [MT-Reading-List](https://github.com/THUNLP-MT/MT-Reading-List)
  * 機械翻訳について、まず読むべき論文とターニングポイントとなった手法の論文などをまとめたリポジトリ。よい研究のガイドとなっている。
* [Evaluating Text Output in NLP: BLEU at your own risk](https://medium.com/@rtatman/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213)
  * 翻訳の評価指標であるBLEUの問題点について述べた記事。主要な論点がまとめられており、とても参考になる。また、派生メトリクスについてもかなりの量が網羅されている。
* [Neural Transfer Learning for Natural Language Processing](http://ruder.io/thesis/)
  * 自然言語処理における転移学習についての、体系的なまとめ。タスクが同じか/異なるかという観点から、同じ場合はDomain Adaptation(言語が異なる場合Cross-lingual)、異なる場合は異なるタスクを同時に学習するか(Multi-Task)・一つずつ学習するか(Sequential)といった形でわけ解説を行なっている。
* [Deep Adversarial Learning for NLP](https://sites.cs.ucsb.edu/~william/papers/AdvNLP-NAACL2019.pdf)
  * 自然言語処理におけるAdversarial Trainingの解説＋研究の紹介を行なっている資料。適用が成功しているのは今の所対話ぐらいだが、まだ可能性は広がっているという内容。
  * [日本語の解説記事](http://www.ai-gakkai.or.jp/my-bookmark_vol34-no5/)
* [NLI with Deep Learning](https://nlitutorial.github.io/)
  * 文関係推論(NLI)に関するデータセットと研究の紹介。SNLIなどで指摘されてるAnnotation Artifact(片方の文だけで推論できるなど)の問題も言及されている。
* [How To Label Data](https://www.lighttag.io/how-to-label-data/)
  * 自然言語処理におけるデータ作成プロセス(アノテーション)をどう進めるのか、ストーリー仕立てで解説している記事。ピザのオーダー受付を自動化するプロジェクトを題材に、目指すべきKPIとの紐付けや精度分析など様々なシーンを解説している。
* [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate)
  * 言語モデルで実際文を生成するときに必要なDecodingについて、各種手法とTransformernによる実装を紹介している記事。単純に一番確率が高い単語をとっていくGreedy Searchから代表的なBeam Searchなどが、実装付きで解説されている。
* [A Visual Survey of Data Augmentation in NLP](https://amitness.com/2020/05/data-augmentation-for-nlp/)
  * 自然言語処理におけるData Augmentationのまとめ。シンプルな言い換え(同義語/分散表現が近い単語への置き換え)以外に、Back Translationや言語モデルによる穴埋め、分散表現のMixupなど比較的最近の手法も紹介されている。
* [CS520: Knowledge Graph](https://web.stanford.edu/class/cs520/)
  * Stanfordで行われているKnowledge Graphの講座。体系的なコースというよりは様々なトピックのアラカルトになっていて、Neo4j(グラフDB)や実用の知識グラフで問題になる不完全Edgeの補完(トリプルのEmbeddingや強化学習で推論する)などの話題が取り上げられている。

### Not Recommend :droplet:

* [深層学習による自然言語処理](https://www.amazon.co.jp/dp/4061529242/)
  * 初心者がこれを読んで学ぶのは厳しい印象。既に理解している人が、定義を確認したり知らなかったモデルをかいつまんでみてみるのに向いている。
* [入門 自然言語処理](https://www.amazon.co.jp/dp/4873114705)
  * 内容がちょっと冗長で、厚さのわりに得るものがあまりなく、わかりやすいとも言い難い内容。

## Audio

* [Neural Nets for Generating Music](https://medium.com/artists-and-machine-intelligence/neural-nets-for-generating-music-f46dffac21c0)
  * 音楽生成の歴史をまとめた大作ブログ。Markov ChainからRNN(LSTM)の利用、Magentaの紹介、CNNを利用したWaveNet、そこからさらにSampleRNNなど2017/8までの音楽生成の歴史と研究を概観できる。
* [Notes on Music Information Retrieval](https://musicinformationretrieval.com/index.html)
  * 音楽の検索を行うためのワークショップの資料。音楽の表現や特徴抽出、またテンポの推定や機械学習による分類方法などがipynb形式で解説されている。

### Additional

* [An Interactive Introduction to Fourier Transforms](http://www.jezzamon.com/fourier/index.html)
  * フーリエ変換をアニメーションで理解する記事。基本的な内容から、3次元、画像への適用などの応用例も紹介されている。
* [Deep Learning for Audio Signal Processing](https://arxiv.org/abs/1905.00078)
  * 音声に対するDNNの適用についてまとめられた資料。音声と画像の性質的な違い(時系列/周波数という相関のない2軸で表現される点、時系列のため順次処理が必要など)を示しその違いを各手法がどう扱っているのかという観点からまとめられている。概要的な資料だが問題設定と手法が上手くまとめられている
* [Generating music in the waveform domain](https://benanne.github.io/2020/03/24/audio-generation.html)
  * 波形ベースの音楽生成のチュートリアル。初歩的な解説から始まり、主要な生成方法(WaveNet、GAN、Glowなど)が全て押さえられており、よくまとまっている。
* [音楽のテンポ分析を行う手法を解説した記事。librosaによる実装も掲載されている。](https://www.wizard-notes.com/entry/music-analysis/tempogram)
  * 音楽のテンポ分析を行う手法を解説した記事。librosaによる実装も掲載されている。

## Reinforcement Learning

* [UCL Course on RL](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html)
  * 最もわかりやすい。強化学習の基礎から知るならこれが一番
* [CS234: Reinforcement Learning Winter 2019](http://web.stanford.edu/class/cs234/index.html)
  * Stanfordの強化学習コース。UCLのコースをベースに作られている。こちらはLecture Noteがついているので、わからない所の補足に使える。
* [6.S094: Deep Learning for Self-Driving Cars](http://selfdrivingcars.mit.edu/)
  * MITの自動運転車のコース。初歩からかなりわかりやすく解説してくれている
* [techcircle_openai_handson](https://github.com/icoxfog417/techcircle_openai_handson)
  * 強化学習で学んだ内容をまとめ、ハンズオン資料にしたもの。
  * Qiitaの記事は[こちら](http://qiita.com/icoxfog417/items/242439ecd1a477ece312)。
  * DQNはこちらの資料も端的にまとまっている。[Human-level control through deep reinforcement learning](http://ir.hit.edu.cn/~jguo/docs/notes/dqn-atari.pdf)
* [Practical_RL](https://github.com/yandexdataschool/Practical_RL)
  * ロシア?の大学で行われている実践的な強化学習の講義の資料(資料は英語なので大丈夫)。資料はかなりしっかりしているほか、OpenAI Gymを利用したDQNの実装なども行っている。
* [Deep Reinforcement Learning and Control Spring 2017, CMU 10703](https://katefvision.github.io/)
* [CS 294: Deep Reinforcement Learning, Spring 2017](http://rll.berkeley.edu/deeprlcourse/)
* [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)
  * Berkeleyで開催された深層強化学習を2日で学ぶブートキャンプの資料。すべての講義の動画とスライドが公開されている。
* [udacity/deep-reinforcement-learning](https://github.com/udacity/deep-reinforcement-learning)
  * Udacityで公開されている深層学習講座にて使用されているサンプルコード。全てJupyter Notebookで参照可能。
  * 他にない所としては、連続値を離散化するテクニックを紹介している。
* [PythonRobotics](https://atsushisakai.github.io/PythonRobotics/)
  * ロボティクスに関わるアルゴリズムと実装が学べるリポジトリ
  * ロボットの操作では強化学習より既存のアルゴリズムが使われることが多いので、それを学ぶのに有用。

### Additional

* [A (Long) Peek into Reinforcement Learning](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#monte-carlo-methods)
  * 速習強化学習ともいうべき記事。これで学ぶというより、学んだあとにポイントを振り返るのに大変便利。
* [Evolutional Strategy](http://blog.otoro.net/2017/10/29/visual-evolution-strategies/)
  * 進化戦略を強化学習に適用する手法について、仕組みと実装を交えながら解説してくれている。
  * 遺伝的アルゴリズムについては、こちらが実装付きで解説を行っている: [Evolution of a salesman: A complete genetic algorithm tutorial for Python](https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35s)
  * 進化戦略/遺伝的アルゴリズムに特化したカンファレンスGECCOのチュートリアルも参考になる([GECCO2019 Tutorial](https://gecco-2019.sigevo.org/index.html/Tutorials))。
* [Deep Reinforcement Learning: An Overview](https://arxiv.org/abs/1701.07274)
  * 2017時点での深層学習による強化学習のまとめ。自然言語処理や音楽生成などの今後の適用領域についての紹介もされており、また教材リストも付属というお得なサマリ
* [Deep Learning for Video Game Playing](https://arxiv.org/pdf/1708.07902.pdf)
  * 深層学習x強化学習でゲームを攻略する研究のまとめ。どんな手法がどんな種類のゲームに使われているかなどもまとめられている。
* [Model-Based Reinforcement Learning](http://people.eecs.berkeley.edu/~cbfinn/_files/mbrl_cifar.pdf)
  * モデルベースの強化学習の解説資料。基本的なところから近年の手法まで解説されている。モデルベースはモデルフリーに比べ資料が少ないため、貴重。
* [Meta-Learning in 50 Lines of JAX](https://blog.evjang.com/2019/02/maml-jax.html)
  * メタラーニングのチュートリアルの記事。メタラーニングは色んな意味で使われているため、まず扱う対象の「メタラーニング」をきちんと定義してくれている。その後、Numpy+勾配計算といった趣のシンプルなライブラリ[JAX](https://github.com/google/jax)を使って実際に実装を行っている。
* [Randomization and the reality gap: how to transfer robotic policies from sim to real](http://josh-tobin.com/assets/pdf/randomization_and_the_reality_gap.pdf)
  * 強化学習 x ロボットの難しさについて解説したスライド。どんなアルゴリズムでも実ロボットの制御と結びついた瞬間に性能が劣化する、データを取ろうにもロボット高い、シミュレーションは正確性に問題あり、という中で複数のシミュレーション学習を複合するマルチタスク/メタラーニングに注目している
  * 強化学習 on ロボットの現場より、という感じでスライド内の文言一つ一つに現場感がある。
* [Quality-Diversity optimisation algorithms](https://quality-diversity.github.io/tutorials)
  * 最適な戦略ただ1つを見つけるのでなく、多様かつ同等のパフォーマンスを持つ戦略を見つけようというQuality-Diversityのチュートリアル。人、動物、昆虫etcが形は違うもののそれぞれ環境に最適化された行動を取るように、最適行動は局所に散在するという考えから新規性/局所最適曲線状の戦略を探索する
* [box2d-lite](https://github.com/erincatto/box2d-lite/blob/master/docs/HowDoPhysicsEnginesWork.pdf)
  * StarCraftを開発するBlizzardの方がまとめた、物理エンジンBox2Dの挙動に関する資料。コード・数式・図の3点セットで説明されており、とてもわかりやすい。
* [実践カルマンフィルタ](https://speakerdeck.com/motokimura/shi-jian-karumanhuiruta)
  * 時系列で変化する状態を推定するカルマンフィルタの解説スライド。カルマンフィルタはロボットなどの自己位置推定に使用されており、実際KITTI(自動運転車)のデータセットでその有効性を確認している(実験コードも公開されている)。
* [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643)
  * 獲得済みの軌跡のみから(環境で新規にデータを取得せずに)学習するオフライン強化学習のチュートリアル資料。問題設定から課題、応用事例まで幅広く解説されている。
  * [日本語解説スライド](https://www.slideshare.net/DeepLearningJP2016/offline-reinforcement-learning-tutorial-review-and-perspectives-on-open-problems)
* [Tutorial on Model-Based Methods in Reinforcement Learning](https://sites.google.com/view/mbrl-tutorial)
  * ICML2020でのモデルベース強化学習の解説スライド。モデルフリーとの違い、モデル構築の手法による違いなど体系的にまとめてくれている。


### Not Recommend :droplet:

* [CS 294: Deep Reinforcement Learning, Spring 2017](http://rll.berkeley.edu/deeprlcourse/)
  * UC Berkeleyの強化学習の講座。講義資料はまとまっているが、数式が多くこれを最初にやるにはきついと思う。
* [Udacity Reinforcement Learning](https://www.udacity.com/course/reinforcement-learning--ud600)
  * 2人の講師の掛け合いで進む講座。テンポはよく時折笑いのポイントもあるが、その分わかりやすいというわけではない。また、ボリュームも少なめ。
* [これからの強化学習](https://www.amazon.co.jp/dp/4627880316)
  * 入門書ではなく論文集に近い体裁なので、学習には不向き
* [速習 強化学習 ―基礎理論とアルゴリズム―](https://www.amazon.co.jp/dp/4320124227)
  * 初心者が速習するための本ではないので注意
  

## Optimization Method

* [An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747)
  * SGDを端緒とした、最適化アルゴリズムについての解説
* [A birds-eye view of optimization algorithms](http://fa.bianp.net/teaching/2018/eecs227at/)
  * 最適化の手法について、図解付きの解説。
* [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
  * Andrej Karpathy先生による、ニューラルネットを学習させるときの心構えと手順についての解説記事。まずデータを徹底的に調べることから開始し、小さく(乱数固定・1バッチへのオーバーフィット確認など)/予測結果を可視化しながら学習を進めていくのが良いとしている。
* [Exploring hyperparameter meta-loss landscapes with Jax](http://lukemetz.com/exploring-hyperparameter-meta-loss-landscapes-with-jax/)
  * JAXを使用したメタラーニングのチュートリアル。非常にシンプルなコードと図で解説されており、進化戦略による最適化についても紹介されている。
* [深層学習の原理を明らかにする理論の試み](https://drive.google.com/file/d/1bNN6VjsgdpJAqxvZ4EKAPpMGq9wfjHqf/view)
  * 従来の理論と矛盾する深層学習の精度を、理論的に説明する研究を紹介したスライド。なぜ多層にするとよいのか、多層でパラメーターが多くなるのになぜ過適合しないのか、どうして学習ができるのか、の3点について近年の研究で明らかになったことがわかりやすく解説されている。

## Probabilistic Modeling

* [Seeing Theory](https://seeing-theory.brown.edu/index.html)
  * 確率/統計についてインタラクティブに学べるサイト。可視化の技法がとてもうまく使われており、抜群にわかりやすい。
* [Introduction to Gaussian Processes - Part I](http://bridg.land/posts/gaussian-processes-1)
  * ガウス過程の解説記事
* [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/)
  * ガウス過程についての解説記事。図、またパラメーターを操作してインタラクティブに試せるコンテンツを用いて解説されており分かりやすい。
* [CSC2541: Scalable and Flexible Models of Uncertainty](https://csc2541-f17.github.io/)
  * トロント大学の確率モデリングの授業。ガウス過程からニューラルネットを使ったベイジアンネットなどを扱い、しかもTensorFlowやStan、Edwardといったライブラリを使った実習もある。
* [Normalizing Flows Tutorial, Part 1: Distributions and Determinants](https://blog.evjang.com/2018/01/nf1.html)
  * 正規分布のようななじみの分布は、簡単にサンプル生成が行えその確からしさも測りやすい。GANやVAEといった生成モデルはこうした解釈性が低いが生成性能はとても高い。シンプルさを保ちつつ生成精度を上げる両取りの方法であるNormalizing Flowについての解説記事。
  * ブログの著者の方による、[Normalizing Flowのチュートリアル資料](https://docs.google.com/presentation/d/1wHJz9Awhlp-PWLZGWJKzF66gzvqdSrhknb-iLFJ1Owo/edit#slide=id.g5b2b1039dc_0_12)
* [Probabilistic-Programming-and-Bayesian-Methods-for-Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)
  * プログラマーのための確率プログラミング、と題した本のコンテンツ。本文と実装が公開されており、Jupyterで動かしながらインタラクティブに学ぶことができる。
  * もとはPyMCだが、TensorFlow probability版も提供されている。

## Causal Inference

* [Introduction to Causal Inference](http://www.ucbbiostat.com/)
  * UC Berkeley 生物統計学部の講義資料。初歩的なところから、実際に利用できるまでが丁寧に解説されている。Rを使った演習問題つき。
* [Introduction to Causal Inference](https://www.bradyneal.com/causal-inference-course)
  * Bengio先生が率いるMILA研究所が開講する因果推論の講座。
  * 必須となるのは基本的な確率の知識のみで、疫学や経済など様々な分野における因果推論を幅広く扱うよう。使用している教科書も公開されている。
* [Causal inference and the data-fusion problem](https://www.pnas.org/content/113/27/7345)
  * 因果推論に関する基本的な内容がまとめられたチュートリアル。異なる出自のデータを統合して推論を行う過程について、課題と課題解決のための手法がきっちりまとめられている。
* [Model-Based Machine Learning](http://www.mbmlbook.com/index.html)
  * 機械学習を利用する際に、問題に対してアルゴリズムを適用するのでなく、問題をモデル化して、それを解くという逆の発想から機械学習の活用を提案している書籍("Model" basedな機械学習としている)。
  * 内容的には推論モデルの本となっていて、Infer.NETによるサンプルコードもあるとのこと。ただ、同じMicrosoftからPythonの因果推論パッケージが出たのでそちらで行ったほうがいいかもしれない([dowhy](https://github.com/Microsoft/dowhy))
  * dowhyについては、因果推論の考え方も含めてまとめられたとても分かりやすい記事がある: [統計的因果推論のためのPythonライブラリDoWhyについて解説：なにができて、なにに注意すべきか](https://www.krsk-phs.com/entry/2018/08/22/060844)
  * 関連して、確率モデルを作成する際のワークフロー[A Principled Bayesian Workflow](https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html)も参考になる
* [Tutorial on Causal Inference and Counterfactual Reasoning](https://causalinference.gitlab.io/kdd-tutorial/)
  * DoWhyを使った、因果推論のチュートリアル(at [KDD2018](https://causalinference.gitlab.io/kdd-tutorial/))。
* [【点と矢印で因果関係を考える】因果関係がないときにデータから関連が生じるパターンとその対策まとめ：因果ダイアグラム（DAG）によるバイアスの視覚的整理](https://www.krsk-phs.com/entry/structural_bias)
  * 「相関関係はあるが因果関係はない」とは具体的にどんなパターンなのかがまとめられた記事。データ分析を行う際はまず目を通しておきたい。
* [ポケモンを題材に因果推論を実践してみる](https://tepppei.hatenablog.com/entry/2020/05/05/113514)
  * モンスターボールとスーパーボール、どちらが捕まえやすいかを因果推論で解き明かす記事。素直にデータを取るとスーパーボールの方が捕獲率が低くなるが、それはなぜか？などポケモントレーナーの行動(本能)に由来するデータの偏りが解明されて行く過程はおもしろくわかりやすい。
* [因果推論のための3ステップ入門](https://www.krsk-phs.com/entry/causalinference_lecture_notes)
  * 因果推論の目的・手法が解説された資料。最終的に知りたいMarginal Effect(母集団全員がAした場合としなかった場合の結果)を、Conditional Effect(性別/地域等条件が入ったデータの結果)から推定するため、等価性の成立を拒む要因の特定(共通因子や選択バイアス)、その調整方法が解説されている。

## Preprocessing/Feature Engineering

* [alicezheng/feature-engineering-book](https://github.com/alicezheng/feature-engineering-book)
  * オライリーの書籍"Feature Engineering for Machine Learning"のサンプルコード集
* Kaggle
  * [Data Cleaning Challenge: Scale and Normalize Data](https://www.kaggle.com/rtatman/data-cleaning-challenge-scale-and-normalize-data): KaggleのKernelを使用して行われている、全5日のデータ前処理講座。欠損値の処理から正規化、文字エンコードの処理などよく使う前処理が実戦形式で学べる。 
  * [My secret sauce to be in top 2% of a kaggle competition](https://towardsdatascience.com/my-secret-sauce-to-be-in-top-2-of-a-kaggle-competition-57cff0677d3c): Kaggleでトップ2%になるために行っているテクニックについての記事。特徴量の分析/可視化に関する手法がメインとなっていて、特徴量分析の重要性を教えてくれる。
* [Real-World Machine Learning](https://www.manning.com/books/real-world-machine-learning)
  * 機械学習を使用するにあたって、データの前処理やモデルの評価方法といった実践的な手法について書かれている本。しかもPython&R対応。
* 自然言語処理
  * [自然言語処理における前処理の種類とその威力](https://qiita.com/Hironsan/items/2466fe0f344115aff177)
  * [自然言語処理の前処理・素性いろいろ](http://yukinoi.hatenablog.com/entry/2018/05/29/120000)
* 画像
  * [画像検索 (特定物体認識) — 古典手法、マッチング、深層学習、Kaggle](https://speakerdeck.com/smly/hua-xiang-jian-suo-te-ding-wu-ti-ren-shi-gu-dian-shou-fa-matutingu-shen-ceng-xue-xi-kaggle)
* [データ分析における特徴量エンジニアリング / feature engineering recipes](https://speakerdeck.com/s_uryu/feature-engineering-recipes)
  * 特徴エンジニアリングの必要性とプロセスがまとまった資料。演習問題もついており、しかもRとPython両方のコードで解説されている。保存版の資料。
  * [解説資料も執筆されている。](https://uribo.github.io/practical-ds/intro.html)

## Engineering

機械学習モデルは作っておしまいではなく、実システムへの組み込みや組み込んだ後の運用も大きな課題となります。また、そもそもどう仕様を決めるのか、と言った点も大きな問題です。それについて学ぶための資料について記載します。

### Product Design

* [役にたちインパクトのある実世界AIを](http://ibisml.org/ibis2018/files/2018/11/kanade.pdf)
  * コンピュータービジョンの大家である、金出先生が語るAI研究のあり方について。AI冬の時代の原因について「人工的に作られた“問題”を“手段”に押し込めようとすることが引き起こした」とし、解けて意味のある具体的な問題に取り組むことが必要としている。
* [スタートアップのための製品要求仕様書(MRD & PRD)の書き方](https://medium.com/@hirokishimada_80077/%E3%82%B9%E3%82%BF%E3%83%BC%E3%83%88%E3%82%A2%E3%83%83%E3%83%97%E3%81%AE%E3%81%9F%E3%82%81%E3%81%AE%E8%A3%BD%E5%93%81%E8%A6%81%E6%B1%82%E4%BB%95%E6%A7%98%E6%9B%B8-mrd-prd-%E3%81%AE%E6%9B%B8%E3%81%8D%E6%96%B9-d5088d795ba5)
  * 作るべきものの決め方についての資料。エレベーターピッチの内容は参考になる。
* [ビジネス、テクノロジー、クリエイティヴの バランスをとるには？](https://note.com/fladdict/n/nebde4365c1b1)
  * サービスの設計にあたって重要なことがとてもシンプルかつ明確に書かれている。
* [クックパッドにおける推薦（と検索）の取り組み](https://speakerdeck.com/chie8842/kutukupatudoniokerutui-jian-tojian-suo-falsequ-rizu-mi)
  * 機械学習に限らず、アルゴリズムをビジネスKPIの向上につなげるための地道な努力がつづられたスライド。
  * チーム作りや、小さい施策のポートフォリオで始めるなど、実践的な話がもりだくさん。
* [Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/research/publication/guidelines-for-human-ai-interaction/)
  * マイクロソフトが発行した、人間とのインタラクションを行う機械学習システムについてのデザインガイドライン。
  * ECサイトや音声アシスタントなど様々なプロダクトから得られたフィードバックを参考に、インタラクションのフェーズごとにガイドラインを設定している。
* [ITエンジニアに易しいUI/UXデザイン](https://www.slideshare.net/ksc1213/ituiux-16732374)
* [Material Design](https://material.io/)
  * Material Designのガイドライン。データの可視化など、機械学習に関わる箇所についても文書があり参考になる。
* [People + AI Guidebook](https://pair.withgoogle.com/)
  * AIシステム開発における、にユーザーとのコミュニケーションの取り方について重点的に書かれた資料。なにをどの程度説明してどう関係を築いていくべきか、かなり実践的な内容。
  * [日本語訳はこちら](http://storywriter.jp/pair/)
* [Explainable Artificial Intelligence (XAI)](https://www.darpa.mil/attachments/XAIProgramUpdate.pdf)
  * DARPAの、説明可能なAIに関する研究の計画資料。えげつないクオリティで書かれていて、研究計画書のお手本ともいえる内容。
* [150 successful machine learning models: 6 lessons learned at Booking.com](https://blog.acolyer.org/2019/10/07/150-successful-machine-learning-models/)
  * Booking.comにおける150の機械学習プロジェクトからえられた6つの知見がまとめられた記事(もとはKDD2019の論文)。活用の費用対効果は平均的に高い点、しかしモデルの精度と価値が比例するとは限らない点、モデル/KPIを早めにテストする必要性など挙げられている
* [Research Methods in Machine Learning](http://web.engr.oregonstate.edu/~tgd/talks/new-in-ml-2019.pdf)
  * 研究テーマの決め方について書かれたスライド、テーマにはライフサイクルがあり、それを意識して選択することが重要(実験的課題設定=>解決策/評価の提示=>改善競争=>課題領域の特定と解決策の対応がとられる=>エンジニアリング(実応用)へ)、など。具体例が豊富に書かれているためイメージが持ちやすい
* [メガヒットゲームのためのゲームデザインパターン２](http://n2-interactive.com/wp/2019/08/17/%E3%83%A1%E3%82%AC%E3%83%92%E3%83%83%E3%83%88%E3%82%B2%E3%83%BC%E3%83%A0%E3%81%AE%E3%81%9F%E3%82%81%E3%81%AE%E3%82%B2%E3%83%BC%E3%83%A0%E3%83%87%E3%82%B6%E3%82%A4%E3%83%B3%E3%83%91%E3%82%BF%E3%83%BC/)
  * ゲームデザインの話ではあるが、サービスにおけるコンセプト開発にとってもとても参考になる内容になっている。
* [どんな機械学習が広告効果を改善するのか](https://speakerdeck.com/qiringji/donnaji-jie-xue-xi-gaguang-gao-xiao-guo-wogai-shan-surufalseka)
  * 広告効果改善のために研究チームとプロダクトチームがタッグを組んで成果を出した軌跡。リーチ率/インプレッション数で有意な効果を出しつつ研究結果がAdKDDに採択されている。
* [Airbnbの機械学習導入から学ぶ](https://speakerdeck.com/pacocat/airbnbfalseji-jie-xue-xi-dao-ru-karaxue-bu)
  * Airbnbでの検索改善の軌跡をまとめたスライド(順次発表された3本の論文が収録されている)。内容が正直すぎて身につまされる思いがするが、その分実践的でとても参考になる。
* [主観と客観を切り替える鍛錬](https://note.com/hebereke/n/n56f6fe99740e)
  * 開発者やデザイナーなど担当としての自分の意見をわきに置いて、ユーザーの客観的な視点に切り替えるために使うフレームワークの紹介。実践的でとても分かりやすい。
* [世界の最大手企業は機械学習を活用したアプリケーションをどのように設計しているか](https://ainow.ai/2021/06/28/256193/)
  * 機械学習を活用している先進的な企業のガイドラインをまとめた記事。各社が機械学習をアプリケーションを適用する際、どのようなステップを踏んでいるのか、何に注意を払っているのかが解説されている。

### Project Planning

* [現代的システム開発概論](https://speakerdeck.com/rtechkouhou/xian-dai-de-sisutemukai-fa-gai-lun)
  * プロジェクトの計画、修正について体系的に書かれた資料。機械学習を使うか否かに関わらず、システム開発を行う前にまず参照しておくべき資料。
* [Google's New Manager Training Slides](https://www.frankmireault.com/blog/googles-new-manager-training-slides)
  * Googleのマネジメント研修で使用されている資料。課題を把握する能力(Cognitive Intelligence)、技術スキルに加えて、自身/チームの感情変化を把握してマネジメント活動に活かすスキル(Emotion Intelligence)が必要とされている。
* [Operationalizing machine learning in processes](https://www.mckinsey.com/business-functions/operations/our-insights/operationalizing-machine-learning-in-processes)
  * 機械学習で業務改善を行うための4ステップをまとめた記事。1.単一の業務ではなくプロセス全体をEnd2Endで改善する方法に焦点を当てる、2.その時に必要な技術要素を選定する、3.OJTのように実環境で検証する、4.機械学習導入プロセスの標準化、のステップとなっている。

### Architecture

* [Web Application Tutorial](https://docs.google.com/presentation/d/1whFnASJKNTblT6o2vF84Cd0j8vhICouXcJAnBdGmMCw/edit?usp=sharing)
  * 基本的なMVCのアーキテクチャとそれを利用する際の注意点について解説しています。
* [Applied machine learning at facebook a datacenter infrastructure perspective (HPCA18)](https://research.fb.com/publications/applied-machine-learning-at-facebook-a-datacenter-infrastructure-perspective/)
  * Facebookで ML as a Service(MLaaS)をどのように提供しており、FB内の機械学習のパイプラインのデザインについて解説している。
  * [解説スライド](https://www.slideshare.net/shunyaueta/applied-machine-learning-at-facebook-a-datacenter-infrastructure-perspective-hpca18)
* [Machine Learning Engineering for Production (MLOps)](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops)
  * MLOpsを学べる講座。解説に加え、TensorFlow Extendedを使用した実践もある。
* [OpenMLOps](https://github.com/datarevenue-berlin/OpenMLOps)
  * OSSを組み合わせてMLOps基盤を構築する方法を解説した記事。実験はJupyterHub、データフローはPrefect、実験管理はMLFlow、モデルホスティングはSeldon、データ管理にFeast、分散実行基盤にDaskを利用している。Data RevenueというAI創薬を手掛ける企業が公開している。

### Team Management

* [Machine Teaching: A New Paradigm for Building Machine Learning Systems](https://arxiv.org/abs/1707.06742)
  * 機械学習を利用したいというニーズに応えていくには、機械学習モデルの構築作業を分業していく必要があるという提言。現在は一人の職人がデータ収集から前処理、モデルの構築まで全部を行い、そのプロセスが属人的になることが多い。なので、最低限アルゴリズム構築と学習は分けようという。
* [2020.06.01 M1勉強会 論文の読み方・書き方・研究室の過ごし方](https://speakerdeck.com/sei88888/2020-dot-06-dot-01-m1mian-qiang-hui-lun-wen-falsedu-mifang-shu-kifang-yan-jiu-shi-falseguo-gosifang)
  * 研究生活を送るためのアドバイス集。中心となる論文を読む/執筆する活動だけでなく、研究室での過ごし方やメンタル・スケジュールコントロールのコツも書かれている。
* [Loon での事後検証: 迅速な開発のための指針](https://cloud.google.com/blog/ja/products/devops-sre/loon-sre-use-postmortems-to-launch-and-iterate)
  * シャットダウンしたLoonから、スタートアップでも早期から導入すべき「事後検証の文化」が公開。インシデントを組織全体に学びを蓄積するチャンスとする文化。スーパーヒーローに頼るのでなく、標準化した学びのプロセスと過失を責めない文化が必要としている。

https://cloud.google.com/blog/ja/products/devops-sre/loon-sre-use-postmortems-to-launch-and-iterate

### Development

* [2021年のエンジニア新人研修の講義資料を公開しました](https://blog.cybozu.io/entry/2021/07/20/100000)
  * サイボウズで使用されている新人研修資料。DevOpsに関連する技術を一通り学ぶことができる。

#### Coding

* [python_exercises](https://github.com/icoxfog417/python_exercises)
  * Pythonのトレーニング用リポジトリです
* [良いコードとは](https://www.slideshare.net/nbykmatsui/ss-55961899)
  * 動けばいいというコードでは、自分の実験の生産性が落ちる可能性があるだけでなく、他の人が再現を行うのも難しくなります。良いコードを書くよう心がけましょう。
* [Writing Code for NLP Research](https://docs.google.com/presentation/d/17NoJY2SnC2UMbVegaRCWA7Oca7UCZ3vHnMqBV4SUayc/edit#slide=id.g44a8796521_0_233)
  * 研究のための開発についてのテクニックについての解説資料(EMNLP2018)
  * 実験(prototyping)か本格的なコンポーネント設計かきちんとわけ、前者ならスクラッチから始めず人のコードを借り修正するところから始めるべし、としている。テストや実験結果記録などにも触れておりかなり実践的。後者で行うべきCIなども解説している。
  * 実際に深層学習系のコードを書く前に、必ず目を通しておきたい。
* [RDBMS in Action](https://speakerdeck.com/saiya_moebius/rdbms-in-action)
  * RDBについての基礎知識。ここに書かれている内容が把握できてればほぼ問題がないと思う。

#### Git

アプリケーションの開発だけでなく、機械学習モデルのソースコードの管理にもバージョン管理ツールは欠かせません。  
ここでは、数あるバージョン管理ツールのうちGitに絞り資料を紹介します。

* [使い始める Git](https://qiita.com/icoxfog417/items/617094c6f9018149f41f)
  * 特定のファイルをバージョン管理対象外にする`.gitignore`は必ず確認しましょう。よく、`.pyc`ファイルや`.ipynb_checkpoints`がリポジトリに入ってしまっている例を見ます。[こちら](https://github.com/github/gitignore)で言語や開発環境に応じたファイルを確認できます。
* [Try Git](https://try.github.io/levels/1/challenges/1)
  * GitHubオフィシャルのGitチュートリアルです

#### Docker

機械学習エンジニアにとってDockerはもはや欠かせないツールになっているので、理解しておくとよいです。

* [コンテナ未経験新人が学ぶコンテナ技術入門](https://www.slideshare.net/KoheiTokunaga/ss-122754942)
  * VMからDocker、Kubernetesに到るまでの過程と周辺技術要素がとてもよくまとめられた資料。この資料だけで、仕組みの理解は済んでしまうと思う。
* [プログラマのためのDocker教科書 インフラの基礎知識＆コードによる環境構築の自動化](https://www.amazon.co.jp/dp/B017UGA7NG)
* [AWSによるクラウド入門](https://tomomano.gitlab.io/intro-aws/#_%E3%81%AF%E3%81%98%E3%82%81%E3%81%AB)
  * 東京大学の計数工学科で使用されているAWS入門の講義資料。科学・エンジニアリングの学生向けで、AWSでGPUの計算を行う方法(Jupyterの立ち上げ方含む)、サーバーレスでのインフラ構築を主に取り上げている。

#### Model Development

* [Rules of Machine Learning: Best Practices for ML Engineering](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf)
  * GoogleのMartinさんが書かれた、機械学習のベストプラクティスについて。単に手順だけでなく、学習が進まないときどうすればいいかなども書かれている。なお、ベストプラクティスその1は「機械学習を使わないことを恐れるな」
* [Best Practices for Applying Deep Learning to Novel Applications](https://arxiv.org/abs/1704.01568)
  * 深層学習をアプリケーションで利用する際のすすめ方や注意点についての話。問題の定義(inとout)をしっかり行うこと、検証が済んでいるモデル(公開されているコードetc)からはじめること、結果の見える化をしとくこと、などが書かれている
  * 読みやすい[Web版が公開された](https://developers.google.com/machine-learning/rules-of-ml/)
* [Applying Deep Learning To Airbnb Search](https://github.com/arXivTimes/arXivTimes/issues/989)
  * Airbnbにおける検索ランキングの改善に、ニューラルネットを適用したプロジェクトの記録資料。
  * どんなモデルから始めるべきか、モデルの性能/状況をどう測るか、特徴エンジニアリングとどう組み合わせるのか、といった機械学習のプロジェクトを進める上で重要な知見が丁寧に書かれている。失敗談についても記載されている。
* [What’s your ML test score? A rubric for ML production systems](https://ai.google/research/pubs/pub45742)
  * 機械学習モデルのテストに関するチェックリスト。特徴量/データセット、モデルの開発・評価プロセス、モデルの運用保守インフラ、パフォーマンス監視の4つの観点でまとめられており、実運用を行う際は3-4ポイントでギリ、5ポイント-の獲得が望ましいとのこと。
* [実用化のための 機械学習の評価尺度の色々](https://docs.google.com/presentation/d/1r1dOVqfQaHh-vyemFO4Vzf5qFMomrf4-CdTYj7OwELE/edit#slide=id.g43c73f7812_0_11)
  * 機械学習活用プロジェクトでよくある「精度が高ければ使える」といった話について、適用業務に合わせた「精度」の測り方をしようという話。
  * 具体例を提示しながら解説されており、イメージがつきやすい。実際にモデルを作る前に目を通しておくと良い。
* [Trends and Our Activities in Quality Management and Assurance of Machine Learning](https://www.slideshare.net/sfchaos/trends-and-our-activities-in-quality-management-and-assurance-of-machine-learning)
  * 機械学習モデルの品質保証についてまとめられたスライド。入力=>出力の再現性確認や、テストデータのカバレッジを担保する手法、またAdversarialへの防衛などについて解説されている。
* [Building ML Pipelines](https://www.buildingmlpipelines.com/)
  * TensorFlowのエコシステムを使用し機械学習パイプラインを構築する方法を書いた本。TensorFlow Extendedの各パーツ紹介以外にApache BEAM/Airflow、Kubeflowによるパイプラインのホスティングについても書かれている。レシピ本の印象で個々のトピックはあまり深くなさそうだが全体感を掴むのによさそう

### Data Collection

#### Scraping

機械学習エンジニアにとって、データを取得するためのスクレイピング技術は欠かせないものになっているため、身に着けておくとよいです。

* [Pythonで始めるウェブスクレイピング実践入門](https://speakerdeck.com/sin_tanaka_21/pythondeshi-meruuebusukureipingushi-jian-ru-men)
* [PythonでWebスクレイピングする時の知見をまとめておく](https://vaaaaaanquish.hatenablog.com/entry/2017/06/25/202924)
  * スクレイピングの技術的な面だけでなく、法律的な面についても触れられています。
  * 著作権法については、2019/1/1からの施行で大幅な緩和が行われます。端的には、スクレイピングで収集したデータを著作権者の害にならない範囲で配布・販売を行うことが可能になります。詳細はこちらをご参考。[改正著作権法が日本のAI開発を加速するワケ　弁護士が解説 ](http://www.itmedia.co.jp/news/articles/1809/06/news017_4.html)。

#### Visualization

* [DataVisualization](https://github.com/neerjad/DataVisualization)
  * 実際のデータを利用した、データ可視化チュートリアル。各種ライブラリ(Seaborn/Bokeh/Plotly/Igraph)ごとに用意されていて使い方を比較できる。
* [Visual Vocabulary](https://gramener.github.io/visual-vocabulary-vega/)
  * データの可視化を行う技法を、可視化したいメトリクス(差なのか共起なのかetc)に応じて分類、紹介してくれているサイト。
* [UW Interactive Data Lab](https://idl.cs.washington.edu/)
  * データの可視化による情報伝達について、実例などをまとめているサイト。
* [from Data to Viz](https://www.data-to-viz.com/)
  * データの形式を選択していくと、それを表すのに適切なグラフを提案してくれるサイト。そのグラフを表示するためのPython/Rのコードも教えてくれる。
* [CS 448B Visualization](https://magrawala.github.io/cs448b-wi20/)
  * Stanford大で行われているデータ可視化の講義(CS 448B)。
  * 空間/インタラクション/アニメーションの使い方、またネットワークやテキストなどデータ構造に応じた可視化の手法が解説されている。実装のサンプル(JavaScriptはObservable、PythonはColab)も提供されている。

### Operation

* [Machine Learning: The High Interest Credit Card of Technical Debt](https://ai.google/research/pubs/pub43146)
  * 機械学習を使い始めた「後」に問題になる点がまとめられた資料。ハイパーパラメーターやデータへの依存、特徴量の劣化/散在に対応する必要性などが書かれている。
* [Dynamic Data Testing](https://medium.com/anomalo-hq/dynamic-data-testing-f831435dba90)
  * データをテストする手法の解説記事。ソフトウェアが稼働する前提となるデータをきちんとテストできていますか?という問いかけよりはじまる。ルールベース(NULLにならない)、固定範囲(x~yの間の値)、動的範囲(95％信頼区間に収まる)、異常検知、の4つに分けて解説している。

### Others

* [AI開発を円滑に進めるための契約・法務・知財](https://www.slideshare.net/hironojumpei/ai-129527593)
  * AI開発にまつわる法律の解釈などをまとめた資料。法律の説明だけでなくケーススタディが掲載されているので(この場合はOKなど)とても参考になる。
* [ライフサイエンス・ヘルスケア業界　未来予想図](https://www2.deloitte.com/jp/ja/pages/life-sciences-and-healthcare/articles/ls/predictions2020.html)
  * ライフサイエンス・ヘルスケア業界の未来像について、当時の数値などから予想を行なった資料。
  * 想像される世界像を提示し、そこで配信されているであろうニュース、具体的なシナリオ、標準化などの世界動向、という形でまとめていて調査資料としてとてもよくできている。

## ESG

### Environment

* [Welcome to the Climate Change AI Wiki](https://wiki.climatechange.ai/wiki/Welcome_to_the_Climate_Change_AI_Wiki)
  * 気候変動問題と機械学習を組み合わせた事例や学習リソースなどをまとめたWikiがオープン。Adaptationのセクションでは、生物多様性や気候変動への対応のため機械学習がどのように使われているか解説されている(リスクアセスメントなど)。

## Others

* [LEARNING WITH LIMITED LABELED DATA: WEAK SUPERVISION AND BEYOND](https://lld-workshop.github.io/#about)
  * NIPS2017で開催された、少数のデータから学習を行う手法のワークショップ
* [From zero to research — An introduction to Meta-learning](https://medium.com/huggingface/from-zero-to-research-an-introduction-to-meta-learning-8e16e677f78a)
  * メタラーニングについて、初歩的なところから解説をしている記事。PyTorchでの実装例も付属している。アニメーションを使った図解が豊富でとても分かりやすい。
* [Hardware Accelerators for Machine Learning (CS 217)](https://cs217.github.io/)
  * 機械学習を効率的に行うためのハードウェア実装を学ぶコースが開講(内容的には演算処理方法に近い)。機械学習の基礎からDNNまでの解説がまずあり、その上で効率的な計算法を学ぶ。もちろんハードウェアとしてFPGAの講義もあり、かなりしっかりしたコースの印象。
* [Towards Open-domain Generation of Programs from Natural Language](http://www.phontron.com/slides/neubig18austin.pdf)
  * プログラムコードの生成に自然言語処理を応用する研究のまとめ。構文の潜在構造をとる、また要約のように抽出と組み合わせる試みなどが取り上げられている。
* [Dealing with Imbalanced Classes in Machine Learning](https://towardsdatascience.com/dealing-with-imbalanced-classes-in-machine-learning-d43d6fa19d2)
  * 機械学習のデータにおいて、ラベル間のサンプル数が不均衡であるケースについての対処法。少数派を増大させるoversampling、多数派を減少させるundersampling、これらより良い少数派を「生成」するSMOTEという手法、また異常検知の問題としてみなしてしまうなど、様々な手法が紹介されている。
* [Troubleshooting Deep Neural Networks](http://josh-tobin.com/troubleshooting-deep-neural-networks)
  * DNNのモデルを構築する際に、上手くいかない場合の対応策をまとめた資料。実装の問題、ハイパーパラメーターの問題、学習の問題、データの問題という4つの観点から解説を行っている。
* [Deep Learning for Anomaly Detection: A Survey](https://arxiv.org/abs/1901.03407)
  * 異常検知に深層学習を使用した研究のサーベイ。既存のサーベイは特定領域にフォーカスしたものが多かったが(動画や医療画像など)、本サーベイでは包括的なまとめを行い、また研究だけでなく産業などでの適用事例についてもまとめている。
 * [A Unifying Review of Deep and Shallow Anomaly Detection](https://arxiv.org/abs/2009.11732)
  * 異常検知 x 深層学習のサーベイ論文をまとめた資料。そもそも異常とは(猫のデータで犬が出てきたら異常だけどスキフトイボブテイルは「外れ値」であり異常でないなど)、という点を定義したうえで各種手法を紹介し統合的なフレームワークも提唱している。ベンチマークの不足といった問題点も挙げている
  * [解説スライド](https://www.slideshare.net/ssuser9eb780/anomaly-detection-survey-239043099)
* [ann-benchmarks](https://github.com/erikbern/ann-benchmarks)
  * クラスタリングなどに用いられる、近似最近傍探索を行なってくれるライブラリをまとめたリポジトリ。理論はわかったので、実際導入したい、という場合にとても参考になる資料。
  * [日本語の解説記事はこちら](https://qiita.com/wasnot/items/20c4f30a529ae3ed5f52)
* [効率的な教師データ作成(アノテーション)のための研究サーベイ](https://tech-blog.abeja.asia/entry/annotation-survery)
  * データセットを作成するアノテーションを効率化するための研究の紹介。画像を中心に既存研究がまとめられている。
* [Recent Advances on Transfer Learning and Related Topics](https://www.slideshare.net/KotaMatsui/recent-advances-on-transfer-learning-and-related-topics)
  * 転移学習の手法とその理論的な解説がなされている資料。
  * 混乱しがちなメタラーニングやFew-shotとの違いについても言及されており、この資料だけで転移学習周りの研究がかなりすっきりと整理できるようになっている。
* [組合せ最適化問題に対する実用的なアルゴリズムとその応用](https://speakerdeck.com/umepon/a-practical-approach-for-hard-combinatorial-optimization-problems-in-real-applications)
  * 制約条件の中で効用を最適化する最適化問題の事例と手法がまとめられた記事。実例が豊富でとても参考になる。
* [深層学習の原理を明らかにする理論の試み](https://drive.google.com/file/d/1bNN6VjsgdpJAqxvZ4EKAPpMGq9wfjHqf/view?fbclid=IwAR0u2MNbv2k9lXDKG6BZPr-G5U5uLS-RMK9gtwa86zcp99q2iVyPJB6ai-0)
  * DNNでなぜうまくいく？の理論的な検証についての解説。
  * なぜ層を増やすと上手くいくのか、パラメーターを増やしても過適合しない・・・どころか精度が上がる理由はなんなのか、という点についてわかりやすい流れで近年の研究が整理されている。
* [Uncertainty Quantification in Deep Learning](https://www.inovex.de/blog/uncertainty-quantification-deep-learning/)
  * 機械学習モデルの不確実性を解説した記事。
  * 出力確率=不確実性ではない点(「わからないもの」を確率50%で出してくれるとは限らない)、不確実性の種類として学習データの不足に起因するもの(epistemic)と出力の揺らぎ(aleatory)の2種類を上げ解説してくれている。
* [Time Series Prediction - A short introduction for pragmatists](https://www.liip.ch/en/blog/time-series-prediction-a-short-comparison-of-best-practices)
  * 時系列予測のチュートリアル記事。基礎的な手法の解説からおためしデータセット、そこでの検証まで紹介されている。基本はシンプルな手法で十分で、Prophetを使えばだいぶ十分という結論。
* [microsoft/forecasting](https://github.com/microsoft/forecasting)
  * 時系列モデルを構築する際のベストプラクティス集。Python/R双方のJupyter Notebookつき。
* [ADVANCES IN FINANCIAL MACHINE LEARNING](https://www.quantresearch.org/Lectures.htm)
  * 金融分野における機械学習を学ぶ講座の資料。ポートフォリオ構築やバックテストなどクオンツ運用への活用が中心になっている。2012年から開講されているようで、各年で行われた講座中の講演資料も掲載されている(新型コロナへの対応もある)。

### Security

* [Differential Privacy Blog Series](https://www.nist.gov/itl/applied-cybersecurity/privacy-engineering/collaboration-space/focus-areas/de-id/dp-blog)
  * NISTが差分プライバシーを解説したブログ記事を公開。技術者だけでなく、事業責任者などプライバシーを意識しなければならない関係者全員で共有できる内容をまとめていくことを目指している。1記事はコンパクトで説明もわかりやすいので、ここから読み進めていくのはお勧め。
* [セキュリティエンジニアのための機械学習入門の入門](https://github.com/13o-bbr-bbq/machine_learning_security/tree/master/Security_and_MachineLearning)
  * セキュリティエンジニアのための機械学習入門記事。
  * 侵入/スパム検知、通信ログの解析について基本的な機械学習手法を適用する方法が解説されている。

### XGBoost

* [A Kaggle Master Explains Gradient Boosting](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)
  * XGBoostをブラックボックスに使っていませんか？ということで、Gradient Boostingの解説。簡単な例からステップをふんでわかりやすく解説している。
* [A Practical Guide to Tree Based Learning Algorithms](https://sadanand-singh.github.io/posts/treebasedmodels/)
  * コンペティションでもよく利用される木構造ベースのアルゴリズム(決定木やランダムフォレストなど)を初歩から学べるコンテンツ
* [Dive into XGBoost](https://speakerdeck.com/hoxomaxwell/dive-into-xgboost)
  * XGBoostへと至る歴史的な背景/手法の進化がまとめられた資料。日本語でのXGBoost解説はあまり見たことがないので、とても貴重。
* [XGBoostのアルゴリズム解説](https://speakerdeck.com/rsakata/santander-product-recommendationfalseapurotitoxgboostfalsexiao-neta?slide=44)

### Domain Specific

* [Deep Learning with Electronic Health Record (EHR) Systems](https://goku.me/blog/EHR)
  * 医療データに対する機械学習の適用についてまとめた記事。医療への適用を考える際には、まず見ておくとよい。
* [メディカルAIコース オンライン講義資料](https://japan-medical-ai.github.io/medical-ai-course-materials/)
  * 医療従事者のためのAI講座資料。CT/MRI画像のセグメンテーションや、心電図の時系列解析といった実践的な内容が解説されている。
* [AI×医用画像の現状と可能性](https://speakerdeck.com/tdys13/aixyi-yong-hua-xiang-falsexian-zhuang-toke-neng-xing)
  * 医療画像に対する機械学習の適用について、近年の研究と市場の動向がまとめられた資料。
  * [2019版も公開された](https://speakerdeck.com/tdys13/aixmedical-imaging-in-japan-2019)
* [How to develop machine learning models for healthcare](https://www.slideshare.net/DeepLearningJP2016/dlhow-to-develop-machine-learning-models-for-healthcare)
  * 医療分野で機械学習を適用する際の全体の流れと各プロセスの注意点がまとめられた資料。医療分野独自の注意点(データの収集におけるプライバシー、モデル評価における対象群の調整etc)についても言及されている。
* [Machine Learning for Healthcare](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-s897-machine-learning-for-healthcare-spring-2019/)
  * 医療のための機械学習のコース。CNNで診断、という技術からのアプローチでなく患者のリスク判定や心臓病、診断チャートへの応用といった医療側の課題から組み立てられており実践的。
  * バイアスや解釈性の問題も扱っており、先進的な内容になっている。
https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-s897-machine-learning-for-healthcare-spring-2019/
* [生命情報向けの機械学習入門](https://github.com/HumanomeLab/mlcourse)
  * ゲノムの解析など生命情報分野の研究における機械学習の活用について解説したリポジトリ。
  * 酵母の細胞周期の同定や、転写因子結合の有無といったタスクを機械学習で予測する方法が紹介されている。
* [初心者向けスポーツ分析チュートリアル「目標達成に導くデータ分析」/ Sports Analysis Tutorial](https://speakerdeck.com/upura/sports-analysis-tutorial)
  * スポーツXデータサイエンスについての資料。単にテクニックだけでなく、アスリート/チームとのコミュニケーション方法などについてまで言及されている。また、iOSのヘルスケアのデータを使った実践チュートリアルもあり、とても参考になる。
* [Coding for Sports Analytics: Resources to Get Started](https://brendankent.com/2020/09/15/coding-for-sports-analytics-resources-to-get-started/)
  * スポーツ解析を行うためのコード集。ラグビーや野球、サッカーやバスケットボールなど様々なスポーツをR/Pythonで解析する実例が集められている。

### Understanding DNN

* [Interpretable Machine Learning](https://hacarus.github.io/interpretable-ml-book-ja/index.html)
  * 解釈可能な機械学習モデルについて解説された書籍の日本語訳。HACARUSの有志の方が翻訳された。口語に近い読みやすい文書で、解説だけでなく実例も掲載されている。
* [Methods for Interpreting and Understanding Deep Neural Networks](https://arxiv.org/abs/1706.07979)
  * DNNの判断を理解するための研究のまとめ。ネットワークが反応する入力を見つける方法(Activation Maximizationなど)、判断根拠となった特徴を入力にマップする方法(Relevance Propagationなど)などを紹介、説明力の比較方法についても記載している
* [Tutorial on Methods for Interpreting and Understanding Deep Neural Networks](http://iphome.hhi.de/samek/pdf/ICASSP2017_T9_3.pdf)
  * ICASSP 2017のチュートリアル資料。ネットワークを逆にたどる形で予測に寄与した点を算出するLayer-wise Relevance Propagationという手法について解説している。
* [Awesome Interpretable Machine Learning](https://github.com/lopusz/awesome-interpretable-machine-learning)
  * DNNを解釈する方法についての論文まとめ。
* [Analysis Methods in Neural Language Processing: A Survey](https://boknilev.github.io/nlp-analysis-methods/)
  * DNN系の自然言語処理のモデルを評価する方法についての体系的なまとめ。モデルの解析による評価(Attentionなど)、モデルのパフォーマンスによる評価(評価セットに対するスコア)、敵対的サンプルによる評価といった3つの観点で研究が整理されている。
* [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)
  * 機械学習において、説明力の高いモデルとモデルを外から解釈(診断)するための手法をまとめた書籍(ただ、ディープ系の話題はあまりカバーしていない)。
  * 販売も行われているが、全文をWebで読むことができる。
* [Explainable AI in Industry (KDD 2019 Tutorial)](https://www.slideshare.net/KrishnaramKenthapadi/explainable-ai-in-industry-kdd-2019-tutorial)
  * KDD2019で行われた、説明可能AIに関するチュートリアル。説明可能性が求められる背景と代表的な手法、また実務での事例が掲載されている。事例が掲載されている資料は珍しい。
* [tidymodels+DALEXによる解釈可能な機械学習 / Tokyo.R83](https://speakerdeck.com/dropout009/tokyo-dot-r83)
  * 機械学習モデルの判断根拠を解析する手法をまとめた記事。タイトルはRのパッケージ紹介となっているが、解釈手法の分類・出力された解釈の用途分類、また各手法の解説/参考文献までまとまっている。解釈手法を導入したい場合の良いスタート地点になると思う。
* [機械学習モデルの判断根拠の説明](https://www.slideshare.net/SatoshiHara3/ver2-225753735)
  * 判断根拠の解釈とは何を意味してどこまで使えるのか、という点から各手法の解説までがまとめられたスライド。説明手法自体への攻撃や危険性などについてもまとめられている。
* [【論文調査】XAI技術の効能を ユーザ実験で評価する研究](https://www.slideshare.net/SatoshiHara3/xai-238616601)
  * 機械学習の判断根拠を説明する手法について、説明された相手(ユーザー)への影響を調査した研究をまとめた資料。モデルの挙動理解・信頼性といった解釈面の効果とユーザー自身の精度向上(いわゆるAugment)の効果などが研究されている。今のところ効果は限定的のよう。

### Adversarial Attack

* [ADVERSARIAL MACHINE LEARNING TUTORIAL](https://aaai18adversarial.github.io/)
  * AAAI2018で開催された、機械学習モデルの識別を誤らせるAdversarialな手法について、攻撃、防衛、検知といったテーマに分けて行われたチュートリアル。
* [Adversarial Examples 分野の動向](https://www.slideshare.net/cvpaperchallenge/adversarial-examples-173590674)
  * Adversarial Exampleの発見から現在までの経緯と今後の展望がまとめられた記事。攻撃/防衛側の果てなき戦いは現在も続いている。資料は画像がメインだが、自然言語・音声でも研究が行われている。
* [機械学習セキュリティのベストプラクティス - Draft NISTIR 8269: A Taxonomy and Terminology of Adversarial Machine Learning -](https://jpsec.ai/nist/sp800-30/guideline/2020/07/08/NISTIR8269.html)
  * 米国のNIST(米国標準技術研究所)で検討されている機械学習モデルのセキュリティ対策をまとめた論文の紹介。既存の研究を「攻撃」「防御」「影響」の3観点に分けて整理している。影響が実際の障害で、信頼性低下・可用性低下・機密性低下などが挙げられている。
* [Adversarial ML Threat Matrix](https://github.com/mitre/advmlthreatmatrix)
  * 機械学習モデルへの攻撃手法を整理したマトリックス。各パターンにおけるケーススタディも提供されている。

### Graph Structure

* [Structured deep models: Deep learning on graphs and beyond](http://tkipf.github.io/misc/SlidesCambridge.pdf)
  * グラフを扱わせたら右に出るものはいない[Thomas Kipf](https://twitter.com/thomaskipf)先生の講義資料。DNNでグラフを扱う際の基本的な考え方からその歴史、最近の研究動向までが網羅されている。
  * グラフは質問回答における推論等に使われる一方、分子構造の推定などにも使われており、多様な分野で応用が広がる熱い分野
* [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)
  * Graph Neural Networkの解説記事。グラフとは何か、グラフデータを扱うタスク、GNNの仕組み、デモ、研究動向まで幅広くまとめられている。特にインタラクティブなコンテンツがさしはさまれており分かりやすい。
  * [Graph Convolutionの解説もあり。](https://distill.pub/2021/understanding-gnns/)
* [Deep Generative Models for Graphs: Methods & Applications](http://i.stanford.edu/~jure/pub/talks2/graph_gen-iclr-may19-long.pdf)
  * グラフ生成に関するチュートリアル資料(ICLR2019)。与えられたデータが持つグラフ構造に似たグラフを生成する方式と、特定性質を持つグラフを生成する方式(化学物質など)の2種類にわけ解説されている。
  * 前者は系列生成(RNN)ベース、後者はGraph Conv+強化学習ベース。
* [グラフ信号処理 ～基礎から応用まで～](https://d65bdd9a-a-61a7baef-s-sites.googlegroups.com/a/msp-lab.org/tanaka/index_j/GSP_MIRU_2019.pdf)
  * グラフ信号処理に関する解説資料。グラフを信号として捉えるとはどういうことか、それに対するフーリエ変換とは?という点がわかりやすく解説されている。
* [生成モデルを中心としたAI創薬最前線](https://speakerdeck.com/elix/elix-cbi-2019)
  * 生成モデルによる分子生成(創薬)について丁寧にまとめられた資料。インプットとなるデータの形式(Fingerprint/SMILES/Graph)、よく使用されるモデル(GAN/RNN/RL/VAE、Graph NN)などについて解説されている。

### Network Optimization

* [機械学習モデルのハイパパラメータ最適化](https://www.slideshare.net/greetech/ss-110811527/)
  * ハイパーパラメータサーチの手法に関するまとめ。サーベイの量に圧倒される

### How to Write / Read

* [松尾ぐみの論文の書き方](http://ymatsuo.com/japanese/ronbun_jpn.html)
  * 論文を書く前に、まずはこちらに目を通しておいた方が良い。
* [Stanford大学流科学技術論文の書き方](http://hontolab.org/tips-for-research-activity/tips-for-writing-technical-papers/)
* [Good Citizen of CVPR](https://www.cc.gatech.edu/~parikh/citizenofcvpr/)
  * CVPRで行われた、良き研究者になるためのガイド的なワークショップの資料。論文の書き方からTodoの管理といった細かいところまで、多くの資料がある。
* [How to Read a Paper](http://blizzard.cs.uwaterloo.ca/keshav/home/Papers/data/07/paper-reading.pdf)
  * 論文の読み方をまとめた論文。読む深さに応じて3段階にわけている。
  * 1. 論文のカテゴリや位置づけ、結論の把握(5~10分)
  * 2. 論文の主張を根拠を基に語れるようになる(~1時間)
  * 3. 著者をトレースして欠けている実験や仮定を発見する(初心者は数時間、ベテランでも1~2時間)
