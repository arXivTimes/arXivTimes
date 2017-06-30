# DataSets

機械学習を行う際に利用可能なデータセットについてまとめています。

# Vision

* [MNIST](http://yann.lecun.com/exdb/mnist/)  
  * 言わずと知れた手書き文字のデータ
* [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)
  * 言わずと知れた10クラス(airplane, automobileなど)にラベル付された画像集。CIFAR-100というより詳細なラベル付けがされたものもある
* [The Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/)
  * CIFAR-10と同様、ラベル付きのデータ。その名の通り動物系
* [YouTube-8M](https://research.google.com/youtube8m/)
  * Googleが公開した800万にも上る動画のデータセット。コンピューター資源が少ない環境でも活用できるように、動画フレームからInception-V3で特徴抽出済みのデータも公開。これがあればTensorFlow(GPU(single))で1日で学習できるらしい。 
* [YouTube-BoundingBoxes](https://research.google.com/youtube-bb/)
  * 物体認識/トラッキングのための大規模なデータセット。YouTubeの動画データを基にしており、時間的に連続した画像に対し物体領域(とクラス)のアノテーションがされている
* [Open Images dataset](https://github.com/openimages/dataset)
  * Googleが公開した900万、ImageNetをこえる6000カテゴリのアノテーション済みデータ。こちらから利用可能。中身はURL+ラベルになっているので、怖がらずにcloneして大丈夫。
* [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  * 20万件の顔画像データと、それに撮影場所や40の特性(笑っているとか、ヒゲがあるとか)がセットになったデータ。また一万件程度は身元のデータがある(有名人などの画像)。
* [MS COCO](http://mscoco.org/home/)
  * 認識・セグメンテーション等のデータを含む物体認識のための統合的なデータセット
  * 画像に対する5つの説明(キャプション)も含む
* [SceneNet RGB-D](https://robotvault.bitbucket.io/scenenet-rgbd.html)
  * 物理シミュレーターでシーン(部屋の中にものが散らばった環境)を作り、そこでカメラの軌跡を設定し映像を作製、その映像のRGB+Depthをデータ化、という感じで生成
  * [SceneNet RGB-D: 5M Photorealistic Images of Synthetic Indoor Trajectories with Ground Truth](https://arxiv.org/abs/1612.05079)
* [ScanNet](http://www.scan-net.org/)
  * 1500以上のスキャンで得られた250万もの3D(RGB-D)画像のデータセット。カメラ位置・サーフェス・セグメンテーションなどのアノテーションがされている。
* [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
  * 物体認識のためのデータセット。MITの[Scene Parsing Challenge](http://sceneparsing.csail.mit.edu/)で使用されている。20,000のセグメンテーション、またさらにその中のパーツといった細かいデータも提供されている。
  * [Semantic Understanding of Scenes through the ADE20K Dataset](https://github.com/arXivTimes/arXivTimes/issues/291)
* [ShapeNet](http://shapenet.cs.stanford.edu/)
  * 3Dモデルのデータセット。家具から飛行機までと、色々な種類のモデルがそろっている。
  * メジャーなカテゴリを集めた[Core55](https://shapenet.cs.stanford.edu/shrec17/)もある
* [ModelNet](http://modelnet.cs.princeton.edu/)
  * シーン/オブジェクト認識のデータセットである[SUN database](http://sun.cs.princeton.edu/)からメジャーなオブジェクトを抜き出して、そのCADイメージを収集したもの。
  * カテゴリ数10のModelNet10と、40のModelNet40がある。
* [SHREC 2014](http://www.itl.nist.gov/iad/vug/sharp/contest/2014/Generic3D/)
  * 既存の3Dモデルのデータセットから、ベンチマークになるような新たなデータセットを構築したもの。
  * PSB/SHREC12GTB/TSB/CCCC/WMB/MSB/BAB/ESBの計8つのデータセットが統合されている(詳細はリンク先のTable1参照)。
  * 最終的には、171クラス8,987モデルのデータセットとなっている。
* [Yobi3D](https://www.yobi3d.com/)
  * フリーの3Dモデル検索エンジン。3Dデータ自体は様々なところから収集されている。データセットという形でまとまってはいないが、用途に合わせて検索し、モデルデータのリンク先を得ることができる。
* [KITTI](http://www.cvlibs.net/datasets/kitti/)
  * 自動運転車のためのデータセット。ドイツの中規模都市であるカールスルーエ周辺～高速道路での運転から得られた画像が提供されている。画像は、最大15台の車と30人の歩行者が映っている。
  * 同様のデータセットに、[ISPRS](http://www.cvlibs.net/projects/autonomous_vision_survey/)、[MOT](https://motchallenge.net/)、[Cityscapes](https://www.cityscapes-dataset.com/)等がある。自動運転は画像認識の複合的なタスクなので、画像でデータがないと思ったら一度目を通してみるといいかもしれない。
* [Mapillary Vistas Dataset](https://www.mapillary.com/dataset/vistas)
  * 自動運転のための、路上画像のデータセット。25,000のアノテーション済みの高解像度データを提供。現在も増え続けており、しかも世界各国の画像が含まれている(日本の道路の画像もある)。
* [quickdraw-dataset](https://github.com/googlecreativelab/quickdraw-dataset)
  * 345カテゴリ、5千万のイラスト描画データ。描画データという名の通り、時系列の筆跡データが提供されている
  * 追加のための学習データセットとして、漢字や羊、象形文字?のデータも公開されている。[sketch-rnn-datasets](https://github.com/hardmaru/sketch-rnn-datasets)
  * 本体のモデルは[こちら](https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn)。
  * [A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477)
* [SpaceNet](https://github.com/SpaceNetChallenge/utilities/tree/master/content/download_instructions)
  * 衛星写真のデータセット。建物の領域などがラベル付けされている
  * データを提供しているリポジトリで、可視化のためのツールなども提供されている
* [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
  * 画像の認識において、単に映っているものを認識するだけでなく、質感などの情報も重要なはず、ということで様々なテクスチャを収集しタグ付したデータセット
  * 全5640画像で、47のカテゴリがアノテーションされている
* [Painter by Numbers(PBN)](https://www.kaggle.com/c/painter-by-numbers/data)
  * 画家・タイトル・画風・ジャンルなどがアノテーションされた画像のデータセット
  * 全23817画像あるが、かなり重たいので(学習用データセットが36GB)アノテーションの配分が均等になるように分けられた13グループのファイル(各1400~2000画像くらい)が用意されている
* [SUN database](http://groups.csail.mit.edu/vision/SUN/)
  * 物体認識・シーン認識のタスクにおいてベースラインとなるようなデータセットを提供している。
  * SUN397: 397のシーンでタグ付けされたデータセット
  * SUN2012: (おそらく2012の)タグ付けがされたデータセット
* [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)
  * 人の動作を検出するためのデータセット。400種類の人間の動作に該当する動画(YouTubeから取得したもの)が、各クラス最低400動画含まれるように構成されている。総件数は30万。
* [20BN-JESTER/ 20BN-SOMETHING-SOMETHING](https://www.twentybn.com/datasets)
  * 20BN-JESTERはハンドジェスチャーの、20BN-SOMETHINGは日常のデバイス操作(コンセント入れたり冷蔵庫のドア閉めたりなど)のデータセットが公開。それぞれ15万、10万と計25万件というボリューム。
* [Manga109](http://www.manga109.org/ja/)
  * 1970年代から2010年代に、実際に出版された109冊の漫画のデータセット。一部の巻についてはページ毎のセリフデータも用意されている。
  * 利用は学術目的限りで、論文などにデータセットの漫画を掲載する際は作者の著作権「©作者名」とManga109の利用であることを明示すること(詳細は上記サイト参照)。
* [AnimeFace Character Dataset](http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/README.html)
  * アニメのキャラクターの顔を集めたデータセット。サイズは160x160で、1キャラクター平均80前後の画像が用意されている


## Visual x NLP

* [VQA](http://www.visualqa.org/index.html)
  * 画像を見て質問に答えるタスクでは、学習した画像についてだけ答えられる、良くある答え(「2つ」とか)を多めに繰り出して精度が上がっているなど明らかな過適合が見られた。そこで真実見たことない画像(Zero-Shot)に回答可能かをテストするためのデータとベースラインモデルの提案 
  * [Zero-Shot Visual Question Answering](https://arxiv.org/abs/1611.05546)
* [CLEVR](http://cs.stanford.edu/people/jcjohns/clevr/)
  * 画像理解のためのデータセット。きちんと理解しているかを診断するために、シンプルな画像(物体がいくつか置いてあるような画像)に対し、様々な内容(物体の色や形といった属性、個数、位置など)を問う質問が用意されている
  * 質問は自然言語の形式だけでなく、プログラムの表現に近い形での形式も用意されている(関数をつなげているような形)。
* [VisDial Dataset](https://visualdialog.org/data)
  * MS COCOの画像をベースに、それについてのQとAが付与されている。Training用に8万、Validation用に4万が利用可能
  * [アノテーションツールまで公開されている](https://github.com/batra-mlp-lab/visdial-amt-chat)ので、さらに集めることも可能。
* [STAIR Captions](https://stair-lab-cit.github.io/STAIR-captions-web/)
  * MS COCOの画像につけられた5つのキャプションについて、日本語でもキャプションをつけたもの(翻訳したわけではなく、独自にアノテートしている)。
  * [STAIR Captions: Constructing a Large-Scale Japanese Image Caption Dataset](https://arxiv.org/abs/1705.00823)
* [Cornell NLVR](http://lic.nlp.cornell.edu/nlvr/)
  * 図と、その図の状態を記述した自然言語のデータセット(具体的には、「少なくとも一つの黒い三角形がある」とか)。
  * [A Corpus of Natural Language for Visual Reasoning](http://alanesuhr.com/suhr2017.pdf)

# NLP

* [日本語対訳データ](http://phontron.com/japanese-translation-data.php?lang=ja)
* [自然言語処理のためのリソース](http://nlp.ist.i.kyoto-u.ac.jp/index.php?NLP%E3%83%AA%E3%82%BD%E3%83%BC%E3%82%B9#g63a7f30)
  * 京都大学から適用されている自然言語処理のためのデータセット。毎日新聞のデータに対する各種言語情報である[京都大学テキストコーパス](http://nlp.ist.i.kyoto-u.ac.jp/index.php?%E4%BA%AC%E9%83%BD%E5%A4%A7%E5%AD%A6%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E3%82%B3%E3%83%BC%E3%83%91%E3%82%B9)(※本文データは別途入手する必要がある)、さまざまなウェブ文書のリード文に対し各種言語情報のアノテーションを行った[京都大学ウェブ文書リードコーパス](http://nlp.ist.i.kyoto-u.ac.jp/index.php?KWDLC)等がある。
* [解析済みブログコーパス](http://nlp.ist.i.kyoto-u.ac.jp/kuntt/)
  * 京都大学と、NTTコミュニケーションの共同研究ユニットで作成されたコーパス。4テーマ（京都観光、携帯電話、スポーツ、グルメ）、249記事、4,186文の解析済みブログコーパス。形態素、構文、格・省略・照応、評判情報がアノテーションされている。
* [livedoor ニュースコーパス](https://www.rondhuit.com/download.html)
  * トピックニュース、Sportsなどの9分野のニュース記事のコーパス
* [青空文庫](http://www.aozora.gr.jp/)
  * 著作権の消滅した作品、また「自由に読んでもらってかまわない」とされたものをテキストとXHTML(一部HTML)形式に電子化した上で揃えている
* [WikiText](https://metamind.io/research/the-wikitext-long-term-dependency-language-modeling-dataset/)
  * 言語モデル学習用のデータセットWikiText-2/WikiText-103の公開。それぞれPenn Treebankよりも2倍&110倍のデータ量。 
* [Sentiment Treebank](https://nlp.stanford.edu/sentiment/code.html)
  * Stanfordの公開している、意味表現ツリーのデータセット
* [Crowdflower](https://www.kaggle.com/crowdflower/datasets)
  * 機械学習プラットフォームのCrowdflowerから提供された、感情タグ付け済みのTwitterデータ。 
* [The Ubuntu Dialogue Corpus](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)
  * Ubuntuのテクニカルサポートの対話データ
* [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
  * Stanfordの公開してる質問応答の大規模データセット
* [Maluuba News QA](http://datasets.maluuba.com/NewsQA)
  * CNNのニュース記事をベースにした質問応答のデータセット。質問数は10万overというサイズ。SQuAD同様、質問はクラウドソーシングで作成しており、回答は本文中の抜粋(一単語とは限らない)になる。しかも、答えられない可能性もあるという歯ごたえのある問題設定になっている。 
* [MS MARCO](http://www.msmarco.org/)
  * Microsoftが公開した質問応答のデータセット(10万件)。質問/回答が、人間のものである点が特徴(Bing=検索エンジンへの入力なのでどこまで質問っぽいかは要確認)。回答はBingの検索結果から抜粋して作成
  * [MS MARCO: A Human Generated MAchine Reading COmprehension Dataset](https://arxiv.org/pdf/1611.09268v1.pdf)
* [TriviaQA: A Large Scale Dataset for Reading Comprehension and Question Answering](http://nlp.cs.washington.edu/triviaqa/)
  * 大規模なQAのデータセット(65万件)。QAだけでなく、Evidence(Answerの根拠となる複数のWebページ、またWikipedia)が付属。
  * 公開時点(2017/5)では、人間の精度80%に対してSQuADで良い成績を収めているモデルでも40%なので、歯ごたえのあるデータセットに仕上がっている。
  * [TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension](http://nlp.cs.washington.edu/triviaqa/docs/triviaQA.pdf)
* [WebQuestions/Free917](https://nlp.stanford.edu/software/sempre/)
  * 5W(When/Where/Who/What/Why)で始まる質問に対する回答を集めたデータセット。
  * WebQuestionsは学習/テスト=3,778/2,032の質問が、Free917は641/276のデータが登録されている
* [TREC QA](http://trec.nist.gov/data/qa.html)
  * 1999年から続く質問回答のタスクで使用されているデータセット。質問はオープンドメインで、回答はクローズドなもの(答えが決まっている(日本の首都は?->東京、のような))
* [HolStep](http://cl-informatik.uibk.ac.at/cek/holstep/)
  * Googleから公開された、論理推論を学習するための大規模データセット。与えられた情報の中で推論に重要な点は何か、各推論間の依存関係、そこから導かれる結論は何か、などといったものがタスクとして挙げられている。
  * [HolStep: A Machine Learning Dataset for Higher-order Logic Theorem Proving](https://arxiv.org/abs/1703.00426)
* [SCONE: Sequential CONtext-dependent Execution dataset](https://nlp.stanford.edu/projects/scone/)
  * Stanfordから公開されている論理推論のためのデータセット。
  * 各シナリオには状況が設定されており(ビーカーがn個ある、絵がn個並んでいる、など)、それに対して5つの連続した操作が自然言語で記述されており(猫の絵を右にずらす、犬の絵をはずす、など)、それらを実行した場合の最後の状態を推定させるのがタスクになる。
  * [Simpler Context-Dependent Logical Forms via Model Projections](https://arxiv.org/abs/1606.05378)
* [DeepMind Q&A Dataset](http://cs.nyu.edu/~kcho/DMQA/)
  * CNN/Daily Mailのデータセット。その記事に関する質問のデータもある
  * Stanfordの研究で、だいぶ簡単な質問しかないことが明らかになっているので利用は注意->[文章を読み、理解する機能の獲得に向けて-Machine Comprehensionの研究動向-](https://www.slideshare.net/takahirokubo7792/machine-comprehension)
* [A Large Self-Annotated Corpus for Sarcasm](https://arxiv.org/pdf/1704.05579.pdf)
  * 皮肉を検出するための大規模コーパスの公開。Redditという掲示板のデータから、130万のデータが提供。アノテーションは投稿者自身が行っている(皮肉コメントには/sがついている)。Redditには皮肉に/sをつける文化があるらしい(HTMLのタグで囲むようにするのが発祥とのこと) 
  * ダウンロードは[こちらから](http://nlp.cs.princeton.edu/SARC/)
* [BookCorpus](http://yknzhu.wixsite.com/mbweb)
  * 10,000以上の書籍のデータのコーパス(ただ、1/4は重複あり)。また、うち11については映画との対応も提供されている(MovieBook dataset)。
  * こちらはさすがに利用申請が必要で、また研究用途のみOK。
* [bAbI](https://research.fb.com/downloads/babi/)
  * Facebook AI Researchが進める自然言語理解のためのプロジェクト(bAbI)で利用されているデータセット
  * 質問応答・対話・言語モデルといった様々なタスクのデータセットが提供されている。


# Audio

* [DCASE](http://www.cs.tut.fi/sgn/arg/dcase2016/task-acoustic-scene-classification)
  * 自然音の分類を行うタスク(公園の音、オフィスの音など)で、学習・評価用データが公開されている。
* [Freesound 4 seconds](https://archive.org/details/freesound4s)
  * FreeSoundの音声データとそのメタデータをまとめたデータセットが公開(普通は頑張ってAPIを叩かないと行けなかった)。音響特徴を捉えるモデルの学習に役立ちそう(以前楽器の分類の学習に使ったことがある)。 
* [AudioSet](https://research.google.com/audioset/)
  * YouTubeから抽出した10秒程度の音に、人の声や車の音といった632のラベル(人の声→シャウト、ささやき、など階層上に定義されている)が付与されている(人手で)。その数その数200万！
* [NSynth Dataset](https://magenta.tensorflow.org/nsynth)
  * 1006の楽器による単音が30万ほど収録されているデータセット

# Knowledge Base

* [Visual Genome](http://visualgenome.org/)
  * 画像とその物体の名前、それらの関係性、またQAなどを含む認識理解に必要なデータを包括しているデータセット
* [Microsoft Concept Graph](https://concept.research.microsoft.com/Home/Introduction)
  * Microfostが公開した、エンティティ間の関係をについてのデータセット。最初はIsA関係(AはBだ的な)のデータで、1,200万のインスタンスと、500万のコンセプト間の、8500万(!)のisA関係を含んでいる。 

# Other

* [MoleculeNet](https://arxiv.org/abs/1703.00564)
  * MoleculeNetという、新薬発見のための分子・分子物理・生体物理・生体？という4種類のデータを包含したデータセットが公開。
  * [DeepChem](https://github.com/deepchem/deepchem)という化学特化のライブラリに組込済
* [Tox21](https://tripod.nih.gov/tox21/challenge/data.jsp)
  * 化学化合物の構造からその毒性(toxic effects)を推定するタスクのためのデータセット。化合物数は12,000、毒性は12の毒性についての値が用意されている。
* [dSPP: Database of structural propensities of proteins](https://peptone.io/dspp)
  * タンパク質(アミノ酸の鎖のベクトル)から構造的傾向スコア(structural propensity score)を予測するためのデータセット。
  * Kerasから使うためのユーティリティも提供されている([dspp-keras](https://github.com/PeptoneInc/dspp-keras))。
* [grocery-shopping-2017](https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2)
  * Instacartという食品のオンラインショップの購買データのデータセットが公開。時間のデータも、時間(0~24時)だけながら利用可能。
* [SARD Dataset](https://samate.nist.gov/SRD/testsuite.php)
  * SARD(Software Assurance Reference Dataset)にて提供されている、ソフトウェアの脆弱性を検証するためのデータセット
  * 脆弱性を含むC/Javaのコードなどが提供されている
* [PHP Security vulnerability dataset](https://seam.cs.umd.edu/webvuldata/data.html)
  * PHPのアプリケーションの脆弱性(CVEのIDなど)と、そのコードから抽出された機械学習で利用する特徴量のデータセット。PHPアプリケーションはPHPMyAdmin、Moodle、Drupalの3点


# Dataset Summary Page

* [kaggle](https://www.kaggle.com/)
  * データ解析のコンペティションサイト。モデルの精度を競い合うことができ、データも提供されている。[Kaggle Datasets](https://www.kaggle.com/datasets)でデータの検索、また公開もできるようになった。
* [国立情報学研究所](http://www.nii.ac.jp/dsc/idr/datalist.html)
  * 日本国内で公開されているデータセットはたいていここを見れば手に入る。ただ研究用途のみで申請書が必要。
* [Harvard Dataverse](https://dataverse.harvard.edu/)
  * ハーバード大学が公開している研究データのセット。自然音のクラス分類のデータ(ESC)などがある。
* [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.html)
  * 機械学習のためのデータセットを集めているサイト。
* [20 Weird & Wonderful Datasets for Machine Learning](https://medium.com/@olivercameron/20-weird-wonderful-datasets-for-machine-learning-c70fc89b73d5)
  * 機械学習で使えるデータセットのまとめ集。UFOレポートとか面白いデータもある。
* [自然言語/音声認識学習用データのまとめ](http://qiita.com/icoxfog417/items/44aeb9486ce1b7130f76)
* [Microsoft Azure Marketplace](https://datamarket.azure.com/browse/data)
  * NFLの試合結果や人口統計など、様々なデータが提供されている(有料なものもあるたが、無料も多い)。
* [ikegami-yukino/dataset-list](https://github.com/ikegami-yukino/dataset-list/blob/master/free_corpus.md)
  * 日本語・英語のテキストコーパスのまとめ
* [beamandrew/medical-data](https://github.com/beamandrew/medical-data)
  * 機械学習のための化学系のデータセットのまとめ

# To make your own

* [ヒューマンコンピュテーションとクラウドソーシング ](https://www.amazon.co.jp/dp/4061529137)
* [Crowdsourcing (for NLP)](http://veredshwartz.blogspot.jp/2016/08/crowdsourcing-for-nlp.html)
  * データを集めるのに欠かせない、クラウドソーシングの活用方法についての記事。クラウドソーシングに向いているタスク、信頼性担保の方法、料金についてなど実践的な内容が紹介されている。
* [Natural Language Annotation for Machine Learning](http://shop.oreilly.com/product/0636920020578.do)
* [バッドデータハンドブック ―データにまつわる問題への19の処方箋](https://www.amazon.co.jp/dp/4873116406)
* [ガラポン](http://garapon.tv/developer/)
  * APIでテレビの字幕のデータを取ることができるらしい
