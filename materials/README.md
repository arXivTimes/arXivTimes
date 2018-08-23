# Materials

機械学習を学ぶための教材

## Machine Learning

* [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning)
  * 最初はここから。ほかのどんな書籍やブログよりよい。
* [fast.ai](http://www.fast.ai/)
  * プログラマーのための機械学習講座といったコース。実践が多く、理論だけでなく手を動かして学びたいという方にお勧め。
* [CSC 321 Winter 2018 Intro to Neural Networks and Machine　Learning](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/)
  * トロント大学のニューラルネット講座。DNNの実装に使用されるフレームワークを意識した解説になっていて、理論的な内容と実装のギャップが少なくなるよう工夫されている。PyTorchのチュートリアルもあり。
* [深層学習 (機械学習プロフェッショナルシリーズ)](https://www.amazon.co.jp/dp/4061529021)
  * ディープラーニングについて良くまとまった書籍
* [Python機械学習プログラミング](https://www.amazon.co.jp/dp/4844380605)
  * scikit-learnを雰囲気で使っていると思ったら参照するとよい書籍。
  * 実践編として、各種アルゴリズムをスクラッチから作る[ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch)もおすすめ。
* [ゼロから作るDeep Learning ――Pythonで学ぶディープラーニングの理論と実装](https://www.oreilly.co.jp/books/9784873117584/)
  * ニューラルネットワークを基礎的な部分から自分で作成していくことで動作について理解できる書籍。ただ、機械学習全般を扱っているわけではないのでその点は注意。
* [How to build a Recurrent Neural Network in TensorFlow](https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767)
  * ゼロから作るRNNといった趣の記事。TensorFlowで素の状態からRNNを作るところから、実際にRNNのモジュールを使って構築するところまでを完全にカバーしている。 

### Additional

* [NIPS HIGHLIGHTS, LEARN HOW TO CODE A PAPER WITH STATE OF THE ART FRAMEWORKS](https://mltrain.cc/events/nips-highlights-learn-how-to-code-a-paper-with-state-of-the-art-frameworks/)
  * NIPS2017で開催された、最新の研究をどうTensorFlowやPyTorchといった機械学習フレームワークのコードに落とし込んでいくのかというワークショップの資料
* [Seedbank](http://tools.google.com/seedbank/)
  * インタラクティブに動かせる機械学習のコードを集めたサイト。事前学習済みのGANモデルの使い方やPerformance RNNによる音楽生成といった応用例から、ニューラルネットワークの仕組みといった基礎的な内容まで幅広い。pandasの使い方などもある。
* [Resources for CS 229 - Machine Learning](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-supervised-learning.html)
  * 機械学習のポイントをまとめたチート集。あれなんだったかな、という時さっと見るのにとても良い。

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
* [畳み込みニューラルネットワークの研究動向](https://www.slideshare.net/ren4yu/ss-84282514)
  * CNNの進化の歴史がまとめられている。これでCNNを学ぶというものではないが、学んだ後背後にある研究系統を体系的に知るには最適な資料。

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
* [[最新版] JSAI2018 チュートリアル「"深層学習時代の" ゼロから始める自然言語処理」](https://www.slideshare.net/yukiarase/jsai2018-101054060/yukiarase/jsai2018-101054060)
  * 人工知能学会2018でのチュートリアル資料。DNN以降の自然言語処理の基礎が入力データ(単語等)の「ベクトル化」であるとして、ではベクトル化の手法としてどんなものがあるのか、といった形で解説がされている。
  * 学習データが少ない時や前処理の必要性などにも触れられている。
* [Document Clustering with Python](http://brandonrose.org/clustering)
  * 自然言語の基本的な特徴量の抽出方法から、各クラスタリングのアルゴリズムを実際試してみるところまでを紹介している。
* [SAILORS 2017 NLP project](https://github.com/abisee/sailors2017)
  * Stanfordで行われている、女子高校生を対象として機械学習/自然言語処理を教えるプログラム[SAILOR](http://ai4all.stanford.edu/)で使用された教材。
  * 学習の手始めとしてもよいが、チュートリアル教材を作る際の参考にもなる。
* [CS 4650 and 7650](https://github.com/jacobeisenstein/gt-nlp-class)
  * ジョージア工科大学の自然言語処理の授業で使われる教科書。扱っているテーマは、完全版と思えるほど幅広い。
  * 図は少なめだが、シンプルな疑似コード的な記述が随所にあり分かりやすく解説されている。

### Information Retrieval

* [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)
  * 情報抽出に関する教科書。わかりやすさに定評のあるCristopher先生の本

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

### Representation

* [Representations for Language: From Word Embeddings to Sentence Meanings](https://nlp.stanford.edu/~manning/talks/Simons-Institute-Manning-2017.pdf)
  * 自然言語の表現学習について、分散表現(word2vec/GloVe)を皮切りに現時点の鉄板であるBidirectional-LSTM + Attention、またCNNの適用などの手法について解説している。この資料だけで、現在の表現学習を概観できる。
  * [Video](https://simons.berkeley.edu/talks/christopher-manning-2017-3-27)
* [From Word to Sense Embeddings: A Survey on Vector Representations of Meaning](https://arxiv.org/abs/1805.04032)
  * 単語分散表現の場合一つの単語は一つのベクトルで表現されるが、本来単語は多様な意味を持つ。そうした「意味」の表現の獲得を目指した研究のサーベイ。教師なし/知識ベースの大きく2つに分けて解説されており、その比較についても記載されている。
* [Deep Learning for Semantic Composition](http://egrefen.com/docs/acl17tutorial.pdf)
  * 文の意味解釈を行うタスクにDNNを適用する際の手法について、体系的なまとめ。

### Others

* [Best Research articles on Deep Learning for Text classification (2015–2016)](https://medium.com/towards-data-science/best-research-articles-on-deep-learning-for-text-classification-2015-2016-aaa7950af775)
  * テキスト分類におけるDNNの適用についてのサーベイ集。RNN/CNNを利用する際は目を通すとよい。
* [Survey of the State of the Art in Natural Language Generation: Core tasks, applications and evaluation](https://github.com/arXivTimes/arXivTimes/issues/563)
* [Deep Learning for Sentiment Analysis : A Survey](https://arxiv.org/abs/1801.07883)
* [Machine Learning on Source Code](https://ml4code.github.io/)
  * 機械学習をシステム開発に役立てる研究のサーベイ。List of Papersには2007年のからの研究がずらりと並ぶ。コードの補完や訂正に関するものが多いが、コミットメッセージやコメントの生成、画面からのコード生成といった珍しいものもある。
* [トピックモデル (機械学習プロフェッショナルシリーズ)](https://www.amazon.co.jp/dp/4061529048)
  * トピックモデルについてはこちらの書籍がとても分かりやすい。
* [All About NLP (AAN)](http://tangra.cs.yale.edu/newaan/)
  * 自然言語処理に関連する論文やチュートリアルなどを集めたポータルサイト。Yale大学が運営しており、共同プロジェクトの実施やデータセットの公開なども行っている。

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

## Reinforcement Learning

* [UCL Course on RL](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html)
  * 最もわかりやすい。強化学習の基礎から知るならこれが一番
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
* [Deep Reinforcement Learning: An Overview](https://arxiv.org/abs/1701.07274)
  * 2017時点での深層学習による強化学習のまとめ。自然言語処理や音楽生成などの今後の適用領域についての紹介もされており、また教材リストも付属というお得なサマリ
* [Deep Learning for Video Game Playing](https://arxiv.org/pdf/1708.07902.pdf)
  * 深層学習x強化学習でゲームを攻略する研究のまとめ。どんな手法がどんな種類のゲームに使われているかなどもまとめられている。


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

## Probabilistic Modeling

* [Introduction to Gaussian Processes - Part I](http://bridg.land/posts/gaussian-processes-1)
  * ガウス過程の解説記事
* [CSC2541: Scalable and Flexible Models of Uncertainty](https://csc2541-f17.github.io/)
  * トロント大学の確率モデリングの授業。ガウス過程からニューラルネットを使ったベイジアンネットなどを扱い、しかもTensorFlowやStan、Edwardといったライブラリを使った実習もある。
* [Normalizing Flows Tutorial, Part 1: Distributions and Determinants](https://blog.evjang.com/2018/01/nf1.html)
  * 正規分布のようななじみの分布は、簡単にサンプル生成が行えその確からしさも測りやすい。GANやVAEといった生成モデルはこうした解釈性が低いが生成性能はとても高い。シンプルさを保ちつつ生成精度を上げる両取りの方法であるNormalizing Flowについての解説記事。

## Preprocessing/Feature Engineering

* [alicezheng/feature-engineering-book](https://github.com/alicezheng/feature-engineering-book)
  * オライリーの書籍"Feature Engineering for Machine Learning"のサンプルコード集
* [Data Cleaning Challenge: Scale and Normalize Data](https://www.kaggle.com/rtatman/data-cleaning-challenge-scale-and-normalize-data)
  * KaggleのKernelを使用して行われている、全5日のデータ前処理講座。欠損値の処理から正規化、文字エンコードの処理などよく使う前処理が実戦形式で学べる。 
* [Real-World Machine Learning](https://www.manning.com/books/real-world-machine-learning)
  * 機械学習を使用するにあたって、データの前処理やモデルの評価方法といった実践的な手法について書かれている本。しかもPython&R対応。
* 自然言語処理
  * [自然言語処理における前処理の種類とその威力](https://qiita.com/Hironsan/items/2466fe0f344115aff177)
  * [自然言語処理の前処理・素性いろいろ](http://yukinoi.hatenablog.com/entry/2018/05/29/120000)
* 画像
  * [画像検索 (特定物体認識) — 古典手法、マッチング、深層学習、Kaggle](https://speakerdeck.com/smly/hua-xiang-jian-suo-te-ding-wu-ti-ren-shi-gu-dian-shou-fa-matutingu-shen-ceng-xue-xi-kaggle)

## Engineering

機械学習モデルは作っておしまいではなく、実システムへの組み込みや組み込んだ後の運用も大きな課題となります。また、そもそもどう仕様を決めるのか、と言った点も大きな問題です。それについて学ぶための資料について記載します。

* [Rules of Machine Learning: Best Practices for ML Engineering](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf)
  * GoogleのMartinさんが書かれた、機械学習のベストプラクティスについて。単に手順だけでなく、学習が進まないときどうすればいいかなども書かれている。なお、ベストプラクティスその1は「機械学習を使わないことを恐れるな」
* [What’s your ML test score? A rubric for ML production systems](https://ai.google/research/pubs/pub45742)
  * 機械学習モデルのテストに関するチェックリスト。特徴量/データセット、モデルの開発・評価プロセス、モデルの運用保守インフラ、パフォーマンス監視の4つの観点でまとめられており、実運用を行う際は3-4ポイントでギリ、5ポイント-の獲得が望ましいとのこと。
* [Machine Teaching: A New Paradigm for Building Machine Learning Systems](https://arxiv.org/abs/1707.06742)
  * 機械学習を利用したいというニーズに応えていくには、機械学習モデルの構築作業を分業していく必要があるという提言。現在は一人の職人がデータ収集から前処理、モデルの構築まで全部を行い、そのプロセスが属人的になることが多い。なので、最低限アルゴリズム構築と学習は分けようという。
* [Best Practices for Applying Deep Learning to Novel Applications](https://arxiv.org/abs/1704.01568)
  * 深層学習をアプリケーションで利用する際のすすめ方や注意点についての話。問題の定義(inとout)をしっかり行うこと、検証が済んでいるモデル(公開されているコードetc)からはじめること、結果の見える化をしとくこと、などが書かれている
  * 読みやすい[Web版が公開された](https://developers.google.com/machine-learning/rules-of-ml/)
* [Machine Learning: The High Interest Credit Card of Technical Debt](https://ai.google/research/pubs/pub43146)
  * 機械学習を使い始めた「後」に問題になる点がまとめられた資料。ハイパーパラメーターやデータへの依存、特徴量の劣化/散在に対応する必要性などが書かれている。
* [Applied machine learning at facebook a datacenter infrastructure perspective (HPCA18)](https://research.fb.com/publications/applied-machine-learning-at-facebook-a-datacenter-infrastructure-perspective/)
  * Facebookで ML as a Service(MLaaS)をどのように提供しており、FB内の機械学習のパイプラインのデザインについて解説している。
  * [解説スライド](https://www.slideshare.net/shunyaueta/applied-machine-learning-at-facebook-a-datacenter-infrastructure-perspective-hpca18)


## Application Development

最近の機械学習界隈では、単純にモデルを作成するだけでなくそれをデモとして仕上げる力も求められている印象があります。  
そのため、アプリケーション開発について学ぶために有用な資料についてもここにまとめておきます。

### Git

アプリケーションの開発だけでなく、機械学習モデルのソースコードの管理にもバージョン管理ツールは欠かせません。  
ここでは、数あるバージョン管理ツールのうちGitに絞り資料を紹介します。

* [使い始める Git](https://qiita.com/icoxfog417/items/617094c6f9018149f41f)
  * 特定のファイルをバージョン管理対象外にする`.gitignore`は必ず確認しましょう。よく、`.pyc`ファイルや`.ipynb_checkpoints`がリポジトリに入ってしまっている例を見ます。[こちら](https://github.com/github/gitignore)で言語や開発環境に応じたファイルを確認できます。
* [Try Git](https://try.github.io/levels/1/challenges/1)
  * GitHubオフィシャルのGitチュートリアルです

### Coding

* [python_exercises](https://github.com/icoxfog417/python_exercises)
  * Pythonのトレーニング用リポジトリです
* [良いコードとは](https://www.slideshare.net/nbykmatsui/ss-55961899)
  * 動けばいいというコードでは、自分の実験の生産性が落ちる可能性があるだけでなく、他の人が再現を行うのも難しくなります。良いコードを書くよう心がけましょう。

### Web

* [Web Application Tutorial](https://docs.google.com/presentation/d/1whFnASJKNTblT6o2vF84Cd0j8vhICouXcJAnBdGmMCw/edit?usp=sharing)
  * 基本的なMVCのアーキテクチャとそれを利用する際の注意点について解説しています。

### Visualization

* [DataVisualization](https://github.com/neerjad/DataVisualization)
  * 実際のデータを利用した、データ可視化チュートリアル。各種ライブラリ(Seaborn/Bokeh/Plotly/Igraph)ごとに用意されていて使い方を比較できる。

### UI/UX

* [ITエンジニアに易しいUI/UXデザイン](https://www.slideshare.net/ksc1213/ituiux-16732374)

### Docker

機械学習エンジニアにとってDockerはもはや欠かせないツールになっているので、理解しておくとよいです。

* [プログラマのためのDocker教科書 インフラの基礎知識＆コードによる環境構築の自動化](https://www.amazon.co.jp/dp/B017UGA7NG)

## Others

* [A Kaggle Master Explains Gradient Boosting](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)
  * XGBoostをブラックボックスに使っていませんか？ということで、Gradient Boostingの解説。簡単な例からステップをふんでわかりやすく解説している。
* [A Practical Guide to Tree Based Learning Algorithms](https://sadanand-singh.github.io/posts/treebasedmodels/)
  * コンペティションでもよく利用される木構造ベースのアルゴリズム(決定木やランダムフォレストなど)を初歩から学べるコンテンツ
* [LEARNING WITH LIMITED LABELED DATA: WEAK SUPERVISION AND BEYOND](https://lld-workshop.github.io/#about)
  * NIPS2017で開催された、少数のデータから学習を行う手法のワークショップ
* [From zero to research — An introduction to Meta-learning](https://medium.com/huggingface/from-zero-to-research-an-introduction-to-meta-learning-8e16e677f78a)
  * メタラーニングについて、初歩的なところから解説をしている記事。PyTorchでの実装例も付属している。アニメーションを使った図解が豊富でとても分かりやすい。
* [Hardware Accelerators for Machine Learning (CS 217)](https://cs217.github.io/)
  * 機械学習を効率的に行うためのハードウェア実装を学ぶコースが開講(内容的には演算処理方法に近い)。機械学習の基礎からDNNまでの解説がまずあり、その上で効率的な計算法を学ぶ。もちろんハードウェアとしてFPGAの講義もあり、かなりしっかりしたコースの印象。
* [Model-Based Machine Learning](http://www.mbmlbook.com/index.html)
  * 機械学習を利用する際に、問題に対してアルゴリズムを適用するのでなく、問題をモデル化して、それを解くという逆の発想から機械学習の活用を提案している書籍("Model" basedな機械学習としている)。
  * 内容的には推論モデルの本となっていて、Infer.NETによるサンプルコードもあるとのこと。ただ、同じMicrosoftからPythonの因果推論パッケージが出たのでそちらで行ったほうがいいかもしれない([dowhy](https://github.com/Microsoft/dowhy))
  * 関連して、確率モデルを作成する際のワークフロー[A Principled Bayesian Workflow](https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html)も参考になる

### Understanding DNN

* [Methods for Interpreting and Understanding Deep Neural Networks](https://arxiv.org/abs/1706.07979)
  * DNNの判断を理解するための研究のまとめ。ネットワークが反応する入力を見つける方法(Activation Maximizationなど)、判断根拠となった特徴を入力にマップする方法(Relevance Propagationなど)などを紹介、説明力の比較方法についても記載している
* [Tutorial on Methods for Interpreting and Understanding Deep Neural Networks](http://iphome.hhi.de/samek/pdf/ICASSP2017_T9_3.pdf)
  * ICASSP 2017のチュートリアル資料。ネットワークを逆にたどる形で予測に寄与した点を算出するLayer-wise Relevance Propagationという手法について解説している。
* [Awesome Interpretable Machine Learning](https://github.com/lopusz/awesome-interpretable-machine-learning)
  * DNNを解釈する方法についての論文まとめ。

### Adversarial Attack

* [ADVERSARIAL MACHINE LEARNING TUTORIAL](https://aaai18adversarial.github.io/)
  * AAAI2018で開催された、機械学習モデルの識別を誤らせるAdversarialな手法について、攻撃、防衛、検知といったテーマに分けて行われたチュートリアル。

### Graph Structure

* [Structured deep models: Deep learning on graphs and beyond](http://tkipf.github.io/misc/SlidesCambridge.pdf)
  * グラフを扱わせたら右に出るものはいない[Thomas Kipf](https://twitter.com/thomaskipf)先生の講義資料。DNNでグラフを扱う際の基本的な考え方からその歴史、最近の研究動向までが網羅されている。
  * グラフは質問回答における推論等に使われる一方、分子構造の推定などにも使われており、多様な分野で応用が広がる熱い分野
