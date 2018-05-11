# Materials

機械学習を学ぶための教材

## Machine Learning

* [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning)
  * 最初はここから。ほかのどんな書籍やブログよりよい。
* [fast.ai](http://www.fast.ai/)
  * プログラマーのための機械学習講座といったコース。実践が多く、理論だけでなく手を動かして学びたいという方にお勧め。
* [CSC 321 Winter 2018 Intro to Neural Networks and Machine　Learning](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/)
  * トロント大学のニューラルネット講座。DNN実装ソフトウェアを意識した解説になっていて、理論的な解説の際も実装が意識されていて、ギャップが少なくなるよう意識されている。PyTorchのチュートリアルもあり。
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

### Not Recommend :droplet:

* [はじめてのパターン認識](https://www.amazon.co.jp/dp/4627849710)
  * 初心者には優しくない。輪講を途中でやめたことがある。
* [パターン認識と機械学習](https://www.amazon.co.jp/dp/4621061224)
  * 読みたい方は、サポート資料として東大松尾研の方々がまとめた輪講用資料を参照するとよい=> [PRML輪読#1~#14](https://www.slideshare.net/matsuolab/)

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
* [Document Clustering with Python](http://brandonrose.org/clustering)
  * 自然言語の基本的な特徴量の抽出方法から、各クラスタリングのアルゴリズムを実際試してみるところまでを紹介している。

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

### Additional

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

## Others

* [A Kaggle Master Explains Gradient Boosting](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)
  * XGBoostをブラックボックスに使っていませんか？ということで、Gradient Boostingの解説。簡単な例からステップをふんでわかりやすく解説している。
* [A Practical Guide to Tree Based Learning Algorithms](https://sadanand-singh.github.io/posts/treebasedmodels/)
  * コンペティションでもよく利用される木構造ベースのアルゴリズム(決定木やランダムフォレストなど)を初歩から学べるコンテンツ
* [LEARNING WITH LIMITED LABELED DATA: WEAK SUPERVISION AND BEYOND](https://lld-workshop.github.io/#about)
  * NIPS2017で開催された、少数のデータから学習を行う手法のワークショップ
* [From zero to research — An introduction to Meta-learning](https://medium.com/huggingface/from-zero-to-research-an-introduction-to-meta-learning-8e16e677f78a)
  * メタラーニングについて、初歩的なところから解説をしている記事。PyTorchでの実装例も付属している。アニメーションを使った図解が豊富でとても分かりやすい。

### Understanding DNN

* [Methods for Interpreting and Understanding Deep Neural Networks](https://arxiv.org/abs/1706.07979)
  * DNNの判断を理解するための研究のまとめ。ネットワークが反応する入力を見つける方法(Activation Maximizationなど)、判断根拠となった特徴を入力にマップする方法(Relevance Propagationなど)などを紹介、説明力の比較方法についても記載している
* [Tutorial on Methods for Interpreting and Understanding Deep Neural Networks](http://iphome.hhi.de/samek/pdf/ICASSP2017_T9_3.pdf)
  * ICASSP 2017のチュートリアル資料。ネットワークを逆にたどる形で予測に寄与した点を算出するLayer-wise Relevance Propagationという手法について解説している。

### Adversarial Attack

* [ADVERSARIAL MACHINE LEARNING TUTORIAL](https://aaai18adversarial.github.io/)
  * AAAI2018で開催された、機械学習モデルの識別を誤らせるAdversarialな手法について、攻撃、防衛、検知といったテーマに分けて行われたチュートリアル。
