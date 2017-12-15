# DataSets

機械学習を行う際に利用可能なデータセットについてまとめています。

# Vision

* [MNIST](http://yann.lecun.com/exdb/mnist/)  
  * 言わずと知れた手書き文字のデータ
* [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)
  * 言わずと知れた10クラス(airplane, automobileなど)にラベル付された画像集。CIFAR-100というより詳細なラベル付けがされたものもある
* [The Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/)
  * CIFAR-10と同様、ラベル付きのデータ。その名の通り動物系
* [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist/blob/master/README.md)
  * ファッション画像のMNIST、を表したデータセット。クラス数はMNISTと同様10クラスで、画像は28x28(グレースケール)、学習：評価データ数は60,000：10,000。
  * MNISTは簡単すぎる、濫用されているといった問題を克服するという側面も意識されている。
* [Open Images dataset](https://github.com/openimages/dataset)
  * Googleが公開した900万、ImageNetをこえる6000カテゴリのアノテーション済みデータ。こちらから利用可能。中身はURL+ラベルになっているので、怖がらずにcloneして大丈夫。
  * リリース後に更新が行われており、2017/11/16にリリースされたV3では370万のバウンディングボックス、970万のラベル付き画像のデータセット(training data)に拡大している。
  * [2017/7/20にbounding boxのデータが追加された](https://research.googleblog.com/2017/07/an-update-to-open-images-now-with.html)。総計約200万で、学習データ中の120万は半自動で付与(人が確認済み)、validationデータ中の80万は人がアノテーションを行なっている。クラス数は600で一部にはラベルもついている。
* [Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)
  * 食品の画像データセット。カテゴリ数は101、画像数は101,000と小粋に数字がそろっている。
  * 各食品は250のテストデータと750の学習用データを持っており、双方人が確認している。画像サイズは全て512pxにリスケールされている
* [Columbia University Image Library (COIL-20)](http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php)
  * 物体を一軸回転させて撮影したデータセット。
  * 20種類の物体を5度刻みで時計回りに回転。画像サイズは128x128。

## Video

* [YouTube-8M](https://research.google.com/youtube8m/)
  * Googleが公開した800万にも上る動画のデータセット。コンピューター資源が少ない環境でも活用できるように、動画フレームからInception-V3で特徴抽出済みのデータも公開。これがあればTensorFlow(GPU(single))で1日で学習できるらしい。 
* [YouTube-BoundingBoxes](https://research.google.com/youtube-bb/)
  * 物体認識/トラッキングのための大規模なデータセット。YouTubeの動画データを基にしており、時間的に連続した画像に対し物体領域(とクラス)のアノテーションがされている
* [Moments in Time Dataset](http://moments.csail.mit.edu/)
  * 3秒間の動画に、何をしているのかがアノテートされたデータセット(文字を書いている、ダイビングをしている、など)。
  * データ数は100万、ラベル数は339で複数付けられる場合もある(歩きながら話している場合、walking+speakingなど)。1ラベルは必ず1000の動画を持つようにしてあり、平均は1757。
* [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)
  * 人の動作を検出するためのデータセット。400種類の人間の動作に該当する動画(YouTubeから取得したもの)が、各クラス最低400動画含まれるように構成されている。総件数は30万。
* [Atomic Visual Actions (AVA)](https://research.google.com/ava/)
  * Googleが公開した人の動作を認識するためのデータセット。YouTubeからの抜粋で作成されており、長さは3秒にそろえられている。
  * 動作ラベルは80、57000の動画に21万件のラベルが付与されている(人単位の付与で人物数は約9万)
  * [AVA: A Video Dataset of Spatio-temporally Localized Atomic Visual Actions](https://arxiv.org/abs/1705.08421)
* [20BN-JESTER/ 20BN-SOMETHING-SOMETHING](https://www.twentybn.com/datasets)
  * 20BN-JESTERはハンドジェスチャーの、20BN-SOMETHINGは日常のデバイス操作(コンセント入れたり冷蔵庫のドア閉めたりなど)のデータセットが公開。それぞれ15万、10万と計25万件というボリューム。
* [TRECVID datasets](http://www-nlpir.nist.gov/projects/trecvid/past.data.table.html)
  * TRECの動画を対象とした情報検索(Video Retrieval)の評価用データセット。
  * 動画中のショット単位(Shot Boundary Detection)、シーン単位(Semantic Indexing)の認識や物体追跡(Instance Search)のためのデータセットが公開されている。
* [UCF101](http://crcv.ucf.edu/data/UCF101.php)
  * 人の動作を検出するためのデータセット。101のクラスを持ち、13,320クリップ、27時間分のデータがある。
  * YouTube のデータを元にしている

## Scene

* [SceneNet RGB-D](https://robotvault.bitbucket.io/scenenet-rgbd.html)
  * 物理シミュレーターでシーン(部屋の中にものが散らばった環境)を作り、そこでカメラの軌跡を設定し映像を作製、その映像のRGB+Depthをデータ化、という感じで生成
  * [SceneNet RGB-D: 5M Photorealistic Images of Synthetic Indoor Trajectories with Ground Truth](https://arxiv.org/abs/1612.05079)
* [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
  * 物体認識のためのデータセット。MITの[Scene Parsing Challenge](http://sceneparsing.csail.mit.edu/)で使用されている。20,000のセグメンテーション、またさらにその中のパーツといった細かいデータも提供されている。
  * [Semantic Understanding of Scenes through the ADE20K Dataset](https://github.com/arXivTimes/arXivTimes/issues/291)
* [Places365](http://places2.csail.mit.edu/download.html)
  * 365カテゴリーのシーンのタグ付け（kitchen、lagoon、yardなど）がされているデータセット
  * 訓練済みモデルが[公開されている](https://github.com/CSAILVision/places365)
  * Plces365は、[Places2](http://places2.csail.mit.edu/) というデータベースのサブセット
* [KITTI](http://www.cvlibs.net/datasets/kitti/)
  * 自動運転車のためのデータセット。ドイツの中規模都市であるカールスルーエ周辺～高速道路での運転から得られた画像が提供されている。画像は、最大15台の車と30人の歩行者が映っている。
  * 同様のデータセットに、[ISPRS](http://www.cvlibs.net/projects/autonomous_vision_survey/)、[MOT](https://motchallenge.net/)、[Cityscapes](https://www.cityscapes-dataset.com/)等がある。自動運転は画像認識の複合的なタスクなので、画像でデータがないと思ったら一度目を通してみるといいかもしれない。
* [Mapillary Vistas Dataset](https://www.mapillary.com/dataset/vistas)
  * 自動運転のための、路上画像のデータセット。25,000のアノテーション済みの高解像度データを提供。現在も増え続けており、しかも世界各国の画像が含まれている(日本の道路の画像もある)。
* [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
  * 画像の認識において、単に映っているものを認識するだけでなく、質感などの情報も重要なはず、ということで様々なテクスチャを収集しタグ付したデータセット
  * 全5640画像で、47のカテゴリがアノテーションされている
* [SUN database](http://groups.csail.mit.edu/vision/SUN/)
  * 物体認識・シーン認識のタスクにおいてベースラインとなるようなデータセットを提供している。
  * SUN397: 397のシーンでタグ付けされたデータセット
  * SUN2012: (おそらく2012の)タグ付けがされたデータセット
* [Team MC^2 : ARC RGB-D Dataset 2017](http://mprg.jp/research/arc_dataset_2017_j)
  * 棚の中におさめられたオブジェクトをロボットに認識、ピッキングさせるAmazon Robotics Challengeのために作られたデータセット。
  * 画像はRGB画像と深度画像の2種類。これに加え、3Dモデルも提供されている
  * アノテーションとして、アイテムごとに色付けしたセグメンテーション画像と、アイテムの四隅の位置(バウンディングボックス)を記録したテキストファイルが提供されている
  * 学習用データは全400シーンx各2回撮影した800枚＋アイテム一つのみ棚に入れた画像(アイテム数40x各10回撮影で400枚、だが公式サイトでは410となっているので何も入っていない棚の画像がある可能性あり)
  * テスト用データは棚にアイテムを入れた全100シーンx各2回撮影の計200枚
* [Matterport3D: Learning from RGB-D Data in Indoor Environments](https://niessner.github.io/Matterport/)
  * 大規模なシーン認識のためのデータセット。90の建物から抽出された10800のパノラマビューに対してオブジェクト/セグメントがアノテーションされている。画像(RGB-D)数は約20万。
  * データセットを扱うためのコードも公開されている([niessner/Matterport](https://github.com/niessner/Matterport))。ここには、ベンチマークも含まれる。
* [3D_Street_View](https://github.com/amir32002/3D_Street_View)
  * Googleのストリートビューから作成したデータセット。同じ地点をカメラ位置を変えて複数回撮影した画像が収録されており、カメラ位置推定や特徴点一致などのタスクに利用できる(118セットで、総画像数は約2500万)。
  * また、都市全体の3Dモデルも提供されている。
* [The German Traffic Sign Detection Benchmark](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset)
  * 道路標識を検知するモデルを学習させるためのデータセット。
  * 認識する道路標識のサイズは16x16から128x128までと幅広く、また様々な光の状態(朝昼夜など)が加味されている。
  * 学習用に900枚、1.6Gというボリューム。またオンラインでのモデル評価システムも完備されている。

## 3D

* [ScanNet](http://www.scan-net.org/)
  * 1500以上のスキャンで得られた250万もの3D(RGB-D)画像のデータセット。カメラ位置・サーフェス・セグメンテーションなどのアノテーションがされている。
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

## Satellite

* [SpaceNet](https://github.com/SpaceNetChallenge/utilities/tree/master/content/download_instructions)
  * 衛星写真のデータセット。建物の領域などがラベル付けされている
  * データを提供しているリポジトリで、可視化のためのツールなども提供されている
* [ABCD (AIST Building Change Detection) dataset](https://github.com/faiton713/ABCDdataset)
  * 津波の被害を受けた建物について、無事だったもの(surviving)と洗い流されたもの(washed-away)それぞれについて津波前後の衛星画像を収めたデータセット。
  * 画像は東日本大震災での被害が対象となっており、建物のダメージの判定は震災後の国土交通省の調査に基づいている。
* [Dublin LiDAR dataset](https://geo.nyu.edu/catalog?f%5Bdct_isPartOf_sm%5D%5B%5D=2015+Dublin+LiDAR)
  * ダブリンの上空からLiDARセンサーで取得した点群のデータセット。300点/m2の密度で、上空以外にも垂直面の情報も提供されているので、3Dモデルを作ることも可能。
  * ダウンロードは範囲ごとになっており、各範囲のページに遷移すると右側の「Tools」の中に「All Downloads」が表示されているので、そこからダウンロードできる。

## BodyParts

* [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  * 20万件の顔画像データと、それに撮影場所や40の特性(笑っているとか、ヒゲがあるとか)がセットになったデータ。また一万件程度は身元のデータがある(有名人などの画像)。
* [MegaFace and MF2: Million-Scale Face Recognition](http://megaface.cs.washington.edu/)
  * 約500万の画像からなる顔のデータセット。70万人分あり、一人当たり平均7画像が用意されている(最小3、最大2500近くとばらつきは結構大きい)
  * 顔を囲ったBounding Boxのデータも併せて提供されている。
* [11k Hands](https://sites.google.com/view/11khands)
  * 様々な年代・性別・肌の色の「手」を集めたデータセット(特に年代は18~75歳と幅広い)。
  * 画像は指を開いたもの/閉じたもの/左右/表裏でそれぞれ取られている。
  * データ総数は11,000件ほど。
* [AISL HDIBPL (Human Depth Images with Body Part Labels) Database](http://www.aisl.cs.tut.ac.jp/database_HDIBPL.html)
  * 深度データから姿勢を推定するためのデータセット。
  * 212x212の深度データに対し、ピクセル単位で10クラスのラベルが付与されている(胴、頭、右上腕、左上腕など・・・)。
* [The Event-Camera Dataset and Simulator](http://rpg.ifi.uzh.ch/davis_data.html)
  * イベントベースのカメラで撮影した動画に対して、実際のモーションキャプチャの情報をセットにしたデータセット。
  * 通常のカメラは一定間隔で画像を撮影するいわゆるパラパラ漫画の方式だが、イベントベースのカメラは画像におけるピクセル変化(イベント)を検出する形のカメラになる。
  * これにより、レイテンシを上げることなく高頻度に変化の検知を行うことができる(ファイルサイズも小さくできる)。[詳細はこちら参照](http://www.rit.edu/kgcoe/iros15workshop/papers/IROS2015-WASRoP-Invited-04-slides.pdf)。
* [MPI Dynamic FAUST(D-FAUST)](http://dfaust.is.tue.mpg.de/)
  * 人体の3次元データに時間を加えた、4次元のモーションデータ(60fpsで撮影)。

## Medical

* [Annotated lymph node CT data](https://wiki.cancerimagingarchive.net/display/Public/CT+Lymph+Nodes)
  * リンパ節の位置がアノテーションされたCT画像。画像数は縦隔90、腹部86。
* [Annotated pancreas CT data](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT)
  * すい臓がアノテーションされた、コントラストを強調した腹部のCT画像。画像数は82。
* [Chest radiograph dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
  * 肺のX線画像データに対して、病名とその位置をアノテーションしたデータセット。30,805人の患者のX線画像112,120枚。
  * [ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf)

## Art

* [Painter by Numbers(PBN)](https://www.kaggle.com/c/painter-by-numbers/data)
  * 画家・タイトル・画風・ジャンルなどがアノテーションされた画像のデータセット
  * 全23817画像あるが、かなり重たいので(学習用データセットが36GB)アノテーションの配分が均等になるように分けられた13グループのファイル(各1400~2000画像くらい)が用意されている
* [quickdraw-dataset](https://github.com/googlecreativelab/quickdraw-dataset)
  * 345カテゴリ、5千万のイラスト描画データ。描画データという名の通り、時系列の筆跡データが提供されている
  * 追加のための学習データセットとして、漢字や羊、象形文字?のデータも公開されている。[sketch-rnn-datasets](https://github.com/hardmaru/sketch-rnn-datasets)
  * 本体のモデルは[こちら](https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn)。
  * [A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477)
* [Manga109](http://www.manga109.org/ja/)
  * 1970年代から2010年代に、実際に出版された109冊の漫画のデータセット。一部の巻についてはページ毎のセリフデータも用意されている。
  * 利用は学術目的限りで、論文などにデータセットの漫画を掲載する際は作者の著作権「©作者名」とManga109の利用であることを明示すること(詳細は上記サイト参照)。
* [eBDtheque](http://ebdtheque.univ-lr.fr/)
  * 漫画のページについて、コマ・フキダシ・フキダシ内のテキストの領域がアノテーションされたデータセット。
  * 漫画はアメリカ、日本、ヨーロッパなど各国のものが集められており、総ページ数は100ページで850コマ。
  * データセットを作るためのアノテーションツールも公開されており、データセットの拡張に貢献できる。
* [AnimeFace Character Dataset](http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/README.html)
  * アニメのキャラクターの顔を集めたデータセット。サイズは160x160で、1キャラクター平均80前後の画像が用意されている
* [GDI (Graphic Design Importance) Dataset](http://www.dgp.toronto.edu/~donovan/layout/index.html)
  * グラフィックデザインのどこに注目するかという、いわゆるヒートマップのデータセット。
  * ぼやけたグラフックを提示し、見たいところをクリックで開けていくという[BubbleView](https://namwkim.github.io/bubbleview/)という手法でデータを取得している
  * 2017/8時点では公式ページでまだ未公開だが、[こちら](https://github.com/cvzoya/visimportance/tree/master/data)から手に入るもよう
* [LLD - Large Logo Dataset](https://data.vision.ee.ethz.ch/cvl/lld/)
  * 50万overの、ロゴ画像のデータセット。Webサイトのfaviconから収集されている。
  * GANでの利用を想定しており、GAN用にロゴっぽくないものなどを除いたClean versionの提供も行われている。
  * 他のロゴ画像のデータセットとしては、Kaggleで公開されている[Favicons](https://www.kaggle.com/colinmorris/favicons)がある。
* [MASSVIS DATASET](http://massvis.mit.edu/)
  * グラフや図といったビジュアライゼーションの効果を検証するためのデータセット。
  * 具体的には、政府の統計、インフォグラフィックス、ニュースや科学雑誌などから抽出したグラフや図に対し、その種類や説明といったものを付与している。
  * 特徴的なのはアイトラッキングのデータで、これにより図表のどこに注目しているかなどを知ることができる。

## Captioning

* [VQA](http://www.visualqa.org/index.html)
  * 画像を見て質問に答えるタスクでは、学習した画像についてだけ答えられる、良くある答え(「2つ」とか)を多めに繰り出して精度が上がっているなど明らかな過適合が見られた。そこで真実見たことない画像(Zero-Shot)に回答可能かをテストするためのデータとベースラインモデルの提案 
  * [Zero-Shot Visual Question Answering](https://arxiv.org/abs/1611.05546)
* [CLEVR](http://cs.stanford.edu/people/jcjohns/clevr/)
  * 画像理解のためのデータセット。きちんと理解しているかを診断するために、シンプルな画像(物体がいくつか置いてあるような画像)に対し、様々な内容(物体の色や形といった属性、個数、位置など)を問う質問が用意されている
  * 質問は自然言語の形式だけでなく、プログラムの表現に近い形での形式も用意されている(関数をつなげているような形)。
* [MS COCO](http://mscoco.org/home/)
  * 認識・セグメンテーション等のデータを含む物体認識のための統合的なデータセット
  * 画像に対する5つの説明(キャプション)も含む
* [COCO-Stuff 10K](https://github.com/nightrome/cocostuff)
  * COCOのデータセット(の一部)にピクセルレベルのアノテーションを行ったデータセットが公開。10,000の画像に91の物体(人や船、象など)がピクセル単位でアノテーションされている。
* [VisDial Dataset](https://visualdialog.org/data)
  * MS COCOの画像をベースに、それについてのQとAが付与されている。Training用に8万、Validation用に4万が利用可能
  * [アノテーションツールまで公開されている](https://github.com/batra-mlp-lab/visdial-amt-chat)ので、さらに集めることも可能。
* [STAIR Captions](https://stair-lab-cit.github.io/STAIR-captions-web/)
  * MS COCOの画像につけられた5つのキャプションについて、日本語でもキャプションをつけたもの(翻訳したわけではなく、独自にアノテートしている)。
  * [STAIR Captions: Constructing a Large-Scale Japanese Image Caption Dataset](https://arxiv.org/abs/1705.00823)
* [Cornell NLVR](http://lic.nlp.cornell.edu/nlvr/)
  * 図と、その図の状態を記述した自然言語のデータセット(具体的には、「少なくとも一つの黒い三角形がある」とか)。
  * [A Corpus of Natural Language for Visual Reasoning](http://alanesuhr.com/suhr2017.pdf)
* [Recipe1M dataset](http://im2recipe.csail.mit.edu/)
  * 100万にも及ぶ、料理画像とそのレシピのペアのデータセット(料理の種類としては80万)。
  * なお、利用は研究用途に限る。


# NLP

* [自然言語処理のためのリソース](http://nlp.ist.i.kyoto-u.ac.jp/index.php?NLP%E3%83%AA%E3%82%BD%E3%83%BC%E3%82%B9#g63a7f30)
  * 京都大学から適用されている自然言語処理のためのデータセット。毎日新聞のデータに対する各種言語情報である[京都大学テキストコーパス](http://nlp.ist.i.kyoto-u.ac.jp/index.php?%E4%BA%AC%E9%83%BD%E5%A4%A7%E5%AD%A6%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E3%82%B3%E3%83%BC%E3%83%91%E3%82%B9)(※本文データは別途入手する必要がある)、さまざまなウェブ文書のリード文に対し各種言語情報のアノテーションを行った[京都大学ウェブ文書リードコーパス](http://nlp.ist.i.kyoto-u.ac.jp/index.php?KWDLC)等がある。
* [解析済みブログコーパス](http://nlp.ist.i.kyoto-u.ac.jp/kuntt/)
  * 京都大学と、NTTコミュニケーションの共同研究ユニットで作成されたコーパス。4テーマ（京都観光、携帯電話、スポーツ、グルメ）、249記事、4,186文の解析済みブログコーパス。形態素、構文、格・省略・照応、評判情報がアノテーションされている。
* [Stanford Rare Word (RW) Similarity Dataset](https://nlp.stanford.edu/~lmthang/morphoNLM/)
  * 文中にあまり登場しない低頻度語について、余りレアではない単語とペアにして、その類似度を付けたデータセット
  * 類似度の評価は、クラウドソーシングで0-10の11段階で評価をしてもらい、つけている。
  * [信頼性について疑問符が付くという報告有り](https://medium.com/@taher.pilevar/is-the-stanford-rare-word-similarity-dataset-a-reliable-evaluation-benchmark-3fe409053011)。低頻度語がどんな語に似ているかは一定の知識が要求されるため、クラウドソーシングには向かないのではないかという指摘もある。
* [日本語単語類似度データセット(JapaneseWordSimilarityDataset)](https://github.com/tmu-nlp/JapaneseWordSimilarityDataset)
  * Stanford Rare Word Similarity Datasetを参考に作成された日本語の単語類似度データセット。
  * 動詞・形容詞・名詞・副詞が対象となっており、クラウドソーシングを利用し10名のアノテータに11段階で単語ペアの類似度をスコアしてもらっている。 
* [WikiText](https://metamind.io/research/the-wikitext-long-term-dependency-language-modeling-dataset/)
  * 言語モデル学習用のデータセットWikiText-2/WikiText-103の公開。それぞれPenn Treebankよりも2倍&110倍のデータ量。 
* [WikiSQL](https://github.com/salesforce/WikiSQL)
  * 自然言語をSQLに対応付けたデータセット。なお、SQLは選択用(SELECT)のみ。
  * 自然言語とSQLの条件・列選択・集計対象列との対応データと、テーブルの定義が提供されている。
* [青空文庫](http://www.aozora.gr.jp/)
  * 著作権の消滅した作品、また「自由に読んでもらってかまわない」とされたものをテキストとXHTML(一部HTML)形式に電子化した上で揃えている
* [青空文庫形態素解析データ集](http://aozora-word.hahasoha.net/index.html)
  * 青空文庫に収録されている作品に対し形態素解析を行ったデータ。CCライセンスで、商用利用も可能。
  * 対象の作品は2012/12時点で公開されており、著作権フラグのない11,176作品。
* [BookCorpus](http://yknzhu.wixsite.com/mbweb)
  * 10,000以上の書籍のデータのコーパス(ただ、1/4は重複あり)。また、うち11については映画との対応も提供されている(MovieBook dataset)。
  * こちらはさすがに利用申請が必要で、また研究用途のみOK。
* [csi-corpus](https://github.com/EdinburghNLP/csi-corpus)
  * 実世界における自然言語タスク、をテーマに作成されたデータセット。具体的には会話から犯人を推定するもので、CSI:科学捜査班の発話文と文中の犯人・容疑者・その他の人、また実際人がそのシーンで思っていた犯人がアノテーションされている
  * なお、実際のドラマのシーンを確認するには当然購入が必要。
  * [Whodunnit? Crime Drama as a Case for Natural Language Understanding](https://arxiv.org/pdf/1710.11601.pdf)

## Parallel Corpus

* [日本語対訳データ](http://phontron.com/japanese-translation-data.php?lang=ja)
* [Tanaka Corpus](http://www.edrdg.org/wiki/index.php/Tanaka_Corpus)
  * 日英翻訳のためのパラレルコーパス。約15万文の日英の分のペアが収録されている。
  * こちらから単語数が4~16である文約5万件を抽出した、単語分割済みのコーパスが別途公開されている([small_parallel_enja](https://github.com/odashi/small_parallel_enja))。
* [JESC: Japanese-English Subtitle Corpus](http://cs.stanford.edu/~rpryzant/jesc/)
  * インターネット上から取得した映画・テレビ番組の字幕に対して作成された日英のパラレルコーパス。
  * 320万文が含まれる
  * [JESC: Japanese-English Subtitle Corpus](https://arxiv.org/abs/1710.10639)

## Classification

* [livedoor ニュースコーパス](https://www.rondhuit.com/download.html)
  * トピックニュース、Sportsなどの9分野のニュース記事のコーパス
* [SemEval-2017 Task 8 RumourEval](http://alt.qcri.org/semeval2017/task8/)
  * 噂の真偽を判定するためのデータセット。審議判定以外に、スタンスの特定のためのデータセットも提供されている。
  * スタンスは特定の情報に対する支持(Support)・否定(Deny)・追加情報の要求(Query)・コメント(Comment)の4種類に分けられている。
* [PHEME rumour dataset: support, certainty and evidentiality](https://www.pheme.eu/2016/06/13/pheme-rumour-dataset-support-certainty-and-evidentiality/)
  * 噂の真偽を判定するためのデータセット。9つのニュースに関連した噂のツイート、及びそれに対する会話が収録されている(会話数が330で、総ツイート数が4,842)。
  * ツイートには、支持(support)・確実性についての確認(certainty)・証拠性についての確認(evidentiality)の3つのスタンスのラベルが付与されている。
* [Noun Compositionality Judgements](https://www.kaggle.com/rtatman/noun-compositionality-judgements)
  * 2語の組み合わせが、逐語的か否か(literal or not literal)をスコアリングしたデータセット。
  * 例えばred appleは赤い・リンゴでそれぞれ独自の意味を持っているが(逐語的)、search engineは「検索エンジン」で一塊の意味となるなど。
* [Enron Email Dataset](https://www.cs.cmu.edu/~enron/)
  * エンロン事件の捜査のさなか、米連邦エネルギー規制委員会(FERC)がインターネット上に公開した実際のエンロン社内のメールデータ。
  * 件数は50万件ほどで、主にエンロンのシニアマネージャーの人達が送ったもの。ユーザー数的には150名ほど。なお添付ファイルは削除されている。
  * メールのデータセットに対して、その意図("要求"か"提案"か)をアノテートしたデータセットが公開されている([EmailIntentDataSet](https://github.com/ParakweetLabs/EmailIntentDataSet))。
* [PubMed 200k RCT dataset](https://github.com/Franck-Dernoncourt/pubmed-rct)
  * 連続する文の分類を行うためのデータセット。具体的には、論文のAbstractに対してこの文は背景、この文は目的、この文は手法・・・といった具合にアノテーションされている。
  * 20万のAbstractに含まれる、230万文にアノテーションが行われている。
  * [PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts](https://arxiv.org/abs/1710.06071)

## Sentiment

* [Sentiment Treebank](https://nlp.stanford.edu/sentiment/code.html)
  * Stanfordの公開している、意味表現ツリーのデータセット
* [Crowdflower](https://www.kaggle.com/crowdflower/datasets)
  * 機械学習プラットフォームのCrowdflowerから提供された、感情タグ付け済みのTwitterデータ。 
* [PersonaBank](https://nlds.soe.ucsc.edu/personabank)
  * 個々人のペルソナを推定するためのコーパスで、個人ブログから抽出された108の個人的なストーリーからなる。
  * 各ストーリーに対しては、意図グラフ(xをyしてzにしようとした、というのがノードとエッジで表現されている)がアノテーションされている。
* [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)
  * MovieLensから取得された、映画のレーティングのデータセット。ユーザー数は6040、3900ほどの映画についての100万のレーティングが提供されている。
  * ユーザー属性(年齢や性別)、映画属性(タイトルなど)、レーティングの3点からなる。推薦システム用のデータセット。
* [MovieTweetings](https://github.com/sidooms/MovieTweetings)
  * IMDBでレートをした時のツイート(`"I rated The Matrix 9/10 http://www.imdb.com/title/tt0133093/ #IMDb"`というような感じの)から収集したデータセット
  * TwitterのユーザーID、映画の情報、レーティングの3点からなる推薦システム用のデータセット。
* [A Large Self-Annotated Corpus for Sarcasm](https://arxiv.org/pdf/1704.05579.pdf)
  * 皮肉を検出するための大規模コーパスの公開。Redditという掲示板のデータから、130万のデータが提供。アノテーションは投稿者自身が行っている(皮肉コメントには/sがついている)。Redditには皮肉に/sをつける文化があるらしい(HTMLのタグで囲むようにするのが発祥とのこと) 
  * ダウンロードは[こちらから](http://nlp.cs.princeton.edu/SARC/)
* [SemEval-2016 Task 5: Aspect-Based Sentiment Analysis](http://alt.qcri.org/semeval2016/task5/)		
  * 単純な極性ではなく、対象と属性(Aspect)を加味したデータセット。		
  * 具体的には、「このパソコンの性能はいまいちだ」という場合、「パソコン#性能, negative」といった具合にアノテーションが行われている		
  * 様々な言語、ドメイン(レストラン、ホテル、家電、電話など)でのデータセットが提供されている。なお日本語はない。
* [Amazon product data](http://jmcauley.ucsd.edu/data/amazon/)
  * Amazonのレビューのデータで、その総数一億四千万。1996年5月から、2014年7月までのレビューが収録されている。
  * データが多すぎるので、各製品にk件以上レビューを持っているユーザーに限定したデータセット、レーティングのみ、またカテゴリごとにデータが提供されている。

## Entity Recognition

* [WNUT17 Emerging and Rare entity recognition](https://noisy-text.github.io/2017/emerging-rare-entities.html)
  * SNSでの投稿などで瞬時に出てくる新語の固有表現を特定するチャレンジ。
  * 人・場所・企業・製品(iPhineとか)・創作物(「君の名は」とか)・グループ(アジカンとか)の計6つのタグがアノテーションされている。
* [WNUT Named Entity Recognition](https://github.com/aritter/twitter_nlp/tree/master/data/annotated/wnut16)
  * TwitterなどのSNSの投稿に対して固有表現を特定するチャレンジ。
* [W-NUT Geolocation Prediction in Twitter](https://noisy-text.github.io/2016/geo-shared-task.html)
  * Twitterの投稿から位置情報を推定するタスク。100万ユーザー分のツイートが収録されている。
  * User-levelとMessage-levelの2種類のタスクが設定されている。
* [Automated Analysis of Cybercriminal Markets](https://evidencebasedsecurity.org/forums/)
  * インターネット上の犯罪取引掲示板のデータセット。取引されているブツに対してアノテーションが行われており、さらに4つの掲示板からデータを取得しているため、異なるドメインでもブツの検知ができるかというドメイン転化の検証にも使用できる。
  * ただ、アノテーション自体はそう多くなく、最大でも700程度(Hack Forums)で最小は80(Blackhat)。

## Knowledge Base

* [Visual Genome](http://visualgenome.org/)
  * 画像とその物体の名前、それらの関係性、またQAなどを含む認識理解に必要なデータを包括しているデータセット
* [Microsoft Concept Graph](https://concept.research.microsoft.com/Home/Introduction)
  * Microfostが公開した、エンティティ間の関係をについてのデータセット。最初はIsA関係(AはBだ的な)のデータで、1,200万のインスタンスと、500万のコンセプト間の、8500万(!)のisA関係を含んでいる。 

## Q&A

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
* [DeepMind Q&A Dataset](http://cs.nyu.edu/~kcho/DMQA/)
  * CNN/Daily Mailのデータセット。その記事に関する質問のデータもある
  * Stanfordの研究で、だいぶ簡単な質問しかないことが明らかになっているので利用は注意->[文章を読み、理解する機能の獲得に向けて-Machine Comprehensionの研究動向-](https://www.slideshare.net/takahirokubo7792/machine-comprehension)
* [QAngaroo](http://qangaroo.cs.ucl.ac.uk/)
  * 文書間のリンクをたどって回答するような質問回答のデータセット。例えば、「スカイツリーは東京にある」「東京は日本の首都である」⇒スカイツリーがある国は？=日本、といった具合。
  * [Constructing Datasets for Multi-hop Reading Comprehension Across Documents](https://arxiv.org/abs/1710.06481)
* [FigureQA](https://datasets.maluuba.com/FigureQA)
  * グラフやプロットといった図に関する質問回答がセットになったデータセット。質問はYes/Noで回答できるものだが、様々なバリエーションが用意されている(XはYより大きいですか、Xは最も小さい値ですか、etc)。また、特徴が偏らないよう質問や図の色が上手く分散するように設計されている。

## Dialog

* [The Ubuntu Dialogue Corpus](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)
  * Ubuntuのテクニカルサポートの対話データ
* [E2E NLG](http://www.macs.hw.ac.uk/InteractionLab/E2E/)
  * End-to-Endの対話システムを構築するためのデータセットが公開。50万発話でが含まれ、ドメインはレストラン検索となっている。発話に対しては固有表現(slot)的なアノテーションもされている(「フレンチが食べたい。500円くらいで」なら、種別=フレンチ、予算=500円など)。
  * [The E2E Dataset: New Challenges For End-to-End Generation](https://arxiv.org/abs/1706.09254)
* [A Multi-Turn, Multi-Domain, Task-Oriented Dialogue Dataset](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/)
  * 複数ドメインにおける、複数ターンの会話のデータセット。
  * 車内のエージェントへの適用を想定しており、スケジュール調整・天気・目的地検索の3つのドメインが用意されている。
  * データ件数は学習用が約2500対話、検証/テスト用がそれぞれ300の総計約3000
* [bAbI](https://research.fb.com/downloads/babi/)
  * Facebook AI Researchが進める自然言語理解のためのプロジェクト(bAbI)で利用されているデータセット
  * 質問応答・対話・言語モデルといった様々なタスクのデータセットが提供されている。
* [self_dialogue_corpus](https://github.com/jfainberg/self_dialogue_corpus)
  * Amazon Alexa Prize用の対話ボットを作成するために収集した、「自己対話」のデータセット。
  * 一人二役を演じる形で作成された対話データセットは、非常に自然でトピックに沿ったものになるので良いとのこと。
  * 約2万5千対話ものデータが提供されている。
  * [Edina: Building an Open Domain Socialbot with Self-dialogues](https://arxiv.org/abs/1709.09816)

## Reasoning

* [HolStep](http://cl-informatik.uibk.ac.at/cek/holstep/)
  * Googleから公開された、論理推論を学習するための大規模データセット。与えられた情報の中で推論に重要な点は何か、各推論間の依存関係、そこから導かれる結論は何か、などといったものがタスクとして挙げられている。
  * [HolStep: A Machine Learning Dataset for Higher-order Logic Theorem Proving](https://arxiv.org/abs/1703.00426)
* [SCONE: Sequential CONtext-dependent Execution dataset](https://nlp.stanford.edu/projects/scone/)
  * Stanfordから公開されている論理推論のためのデータセット。
  * 各シナリオには状況が設定されており(ビーカーがn個ある、絵がn個並んでいる、など)、それに対して5つの連続した操作が自然言語で記述されており(猫の絵を右にずらす、犬の絵をはずす、など)、それらを実行した場合の最後の状態を推定させるのがタスクになる。
  * [Simpler Context-Dependent Logical Forms via Model Projections](https://arxiv.org/abs/1606.05378)

## Summarization/Correction

* [DUC 2004](http://www.cis.upenn.edu/~nlp/corpora/sumrepo.html)
  * 文章要約のためのデータセット。ベースラインとなるアルゴリズムによる要約結果も収録されており、それらのROUGEスコアと比較が可能。
* [boxscore-data](http://lstm.seas.harvard.edu/docgen/)
  * バスケットボールの試合のスコアと、試合結果についての要約をペアとしたデータセット。数値表現をテキストにする試み。
  * Rotowire/SBNationといったスポーツサイトからデータを収集しており、総計約15000のペアが収録されている。
* [AESW](http://textmining.lt/aesw/index.html)
  * 文書校正前後の論文を集めたデータセット。
  * 学習データには約100万文が含まれ、そのうち46万件ほどに校正による修正が入っている。
* [Lang-8 dataset](http://cl.naist.jp/nldata/lang-8/)
  * 語学学習を行うSNSであるLang-8から収集されたデータセット。Lang-8では学習している言語で作文を行うと、その言語を母国語としている人から添削を受けることができる。この学習者の作文と訂正された作文のペアがデータセットとして収録されている。
  * 10言語のデータが含まれており、総数は約58万文書に及ぶ。
  * 実はNAISTが公開しており、詳細はこちらから参照できる。[語学学習 SNS の添削ログからの母語訳付き学習者コーパスの構築に向けて](https://www.ninjal.ac.jp/event/specialists/project-meeting/files/JCLWorkshop_no6_papers/JCLWorkshop_No6_27.pdf)


# Audio

* [DCASE](http://www.cs.tut.fi/sgn/arg/dcase2016/task-acoustic-scene-classification)
  * 自然音の分類を行うタスク(公園の音、オフィスの音など)で、学習・評価用データが公開されている。
* [Freesound 4 seconds](https://archive.org/details/freesound4s)
  * FreeSoundの音声データとそのメタデータをまとめたデータセットが公開(普通は頑張ってAPIを叩かないと行けなかった)。音響特徴を捉えるモデルの学習に役立ちそう(以前楽器の分類の学習に使ったことがある)。 
* [AudioSet](https://research.google.com/audioset/)
  * YouTubeから抽出した10秒程度の音に、人の声や車の音といった632のラベル(人の声→シャウト、ささやき、など階層上に定義されている)が付与されている(人手で)。その数その数200万！
* [NSynth Dataset](https://magenta.tensorflow.org/nsynth)
  * 1006の楽器による単音が30万ほど収録されているデータセット
* [Yamaha e-Piano Competition dataset](http://www.piano-e-competition.com/midi_2004.asp)
  * 公式にデータセットとして配布されているわけではないが、YAMAHAのジュニアコンペティションで実際に演奏されたピアノのMIDIデータが公開されている。[Performance RNN](https://magenta.tensorflow.org/performance-rnn)で使用されていたデータセット。
* [The Largest MIDI Collection on the Internet](https://www.reddit.com/r/WeAreTheMusicMakers/comments/3ajwe4/the_largest_midi_collection_on_the_internet/)
  * 公開されているMIDIデータを収集した大規模なデータセット(※もちろん有料なコンテンツは含まれない)。
  * ポップ、クラシック、ゲーム音楽など多彩なジャンルで構成されており、総ファイル数13万・約100時間分のデータとなっている。
  * Tronto大学の[Song From PI](http://www.cs.toronto.edu/songfrompi/)で使用されたデータセット
* [The MagnaTagATune Dataset](http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)
  * [TagATune](http://www.cs.cmu.edu/~elaw/papers/tagatune.pdf)というゲームを通じて音楽データを収集する試みにより、[Magnatune](http://magnatune.com/)という音楽サイトから収集されたデータセット。
  * 約3万曲に対し、音楽ジャンルなどのタグが付与されている。
* [声優統計コーパス](http://voice-statistics.github.io/)
  * 独自に構築された音素バランス文を、プロの女性声優3名が読み上げたものを録音したコーパス。
  * 3パターンの感情(通常・喜び・怒り)での読み上げが含まれる。48kHz/16bitのWAVファイルで、総長約2時間、総ファイルサイズ720MB。
* [JSUT(Japanese speech corpus of Saruwatari Lab, University of Tokyo)](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)
  * 日本語テキストと読み上げ音声からなるコーパス。一人の日本語女性話者の発音を無響室で録音。録音時間は10時間で、サンプリングレートは48kHz。
  * 常用漢字の音読み/訓読みを全てカバーするといった網羅性だけでなく、旅行ドメインのフレーズといったドメイン特化のものも収録されている。
* [Speech Commands Dataset](https://www.tensorflow.org/versions/master/tutorials/audio_recognition)
  * TensorFlowとAIYのチームから公開された、30種類のYes, No, Up, Downなどといった短い音声による指示/応答を集めたデータセット。総数は65,000。
  * このデータセットを利用した音声認識モデルの構築手順が、TensorFlowのチュートリアルとして提供されている。
* [The Spoken Wikipedia Corpora](http://nats.gitlab.io/swc/)
  * Wikipediaの記事を読み上げたデータセット。音声と単語の対応、単語と文中語の対応がアノテーションされている(単語がfive, hundredだった場合、文中語の500に対応、など)。
  * しかも多言語のデータセットで、英語・ドイツ語・オランダ語が提供されている。
* [Common Voice](https://voice.mozilla.org/data)
  * Mozillaが公開した、音声認識のためのデータセット。音声データは500時間分、2万人以上から録音という世界で二番目の規模。
  * モデルも公開されている: [DeepSpeech](https://github.com/mozilla/DeepSpeech)

# Other

* [MoleculeNet](https://arxiv.org/abs/1703.00564)
  * MoleculeNetという、新薬発見のための分子・分子物理・生体物理・生体？という4種類のデータを包含したデータセットが公開。
  * [DeepChem](https://github.com/deepchem/deepchem)という化学特化のライブラリに組込済
* [Tox21](https://tripod.nih.gov/tox21/challenge/data.jsp)
  * 化学化合物の構造からその毒性(toxic effects)を推定するタスクのためのデータセット。化合物数は12,000、毒性は12の毒性についての値が用意されている。
* [dSPP: Database of structural propensities of proteins](https://peptone.io/dspp)
  * タンパク質(アミノ酸の鎖のベクトル)から構造的傾向スコア(structural propensity score)を予測するためのデータセット。
  * Kerasから使うためのユーティリティも提供されている([dspp-keras](https://github.com/PeptoneInc/dspp-keras))。
* [MIMIC](https://mimic.physionet.org/)
  * 40,000人のケアが必要な重篤な患者についてのデータセット。人口統計、バイタルサイン、検査結果、医薬品情報などが含まれる。
  * 利用にあたってはまずCITIの"Data or Specimens Only Research"というオンライン講座を受講する必要がある([こちら](https://mimic.physionet.org/gettingstarted/access/)参照)。
* [grocery-shopping-2017](https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2)
  * Instacartという食品のオンラインショップの購買データのデータセットが公開。時間のデータも、時間(0~24時)だけながら利用可能。
* [SARD Dataset](https://samate.nist.gov/SRD/testsuite.php)
  * SARD(Software Assurance Reference Dataset)にて提供されている、ソフトウェアの脆弱性を検証するためのデータセット
  * 脆弱性を含むC/Javaのコードなどが提供されている
* [PHP Security vulnerability dataset](https://seam.cs.umd.edu/webvuldata/data.html)
  * PHPのアプリケーションの脆弱性(CVEのIDなど)と、そのコードから抽出された機械学習で利用する特徴量のデータセット。PHPアプリケーションはPHPMyAdmin、Moodle、Drupalの3点
* [気象庁：過去の気象データ](http://www.data.jma.go.jp/gmd/risk/obsdl/index.php)
  * 地点毎になるが(複数選択可能)、過去の気象データをCSV形式でダウンロードできる。
* [Global Terrorism Database](https://www.kaggle.com/START-UMD/gtd)
  * 1970~2016年(なぜか2013年だけない)の間の世界で発生したテロ事件のデータセット。その数17万件。
  * STARTというテロ対策を研究する団体がメンテナンスを行っており、特徴量として発生した場所や犯人や手口などが含まれている。
* [GoGoD](http://senseis.xmp.net/?GoGoDCD)
  * プロの囲碁棋士の対局データセット。85,000局分が含まれており、お値段は15USD
* [wangjinzhuo/pgd](https://github.com/wangjinzhuo/pgd)
  * プロの囲碁棋士の対局データセット。GitHub上でフリーで公開されており、約25万局が収録されている。
* [TorchCraft/StarData](https://github.com/TorchCraft/StarData)
  * StarCraftのプレイデータ。約6万5千プレイ、フレーム数にして15億(!!)という大規模なデータセット。
* [THE STANFORD OPEN POLICING PROJECT](https://openpolicing.stanford.edu/)
  * スタンフォードが取り組んでいる、法的機関(警察)による交通取り締まりのオープンデータプロジェクト。
  * データには、違反日時や場所、ドライバーの年齢や性別などが記録されている(Jupyterのサンプルも公開されている)。

# Dataset Summary Page

* [kaggle](https://www.kaggle.com/)
  * データ解析のコンペティションサイト。モデルの精度を競い合うことができ、データも提供されている。[Kaggle Datasets](https://www.kaggle.com/datasets)でデータの検索、また公開もできるようになった。
* [人文学オープンデータ共同利用センター](http://codh.rois.ac.jp/)
  * 日本の古典(徒然草や源氏物語)の書籍画像、また本文テキストなどのデータを公開しているサイト。中にはレシピ本などの面白いものもある。
  * 機械学習への応用をきちんと想定しており、古文字の画像認識用データセットなども公開している。
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
* [Web Technology and Information Systems](https://www.uni-weimar.de/en/media/chairs/computer-science-and-media/webis/corpora/)
  * Web Technology and Information Systemsの研究で使用されたコーパス集


# To make your own

* [ヒューマンコンピュテーションとクラウドソーシング ](https://www.amazon.co.jp/dp/4061529137)
* [Crowdsourcing (for NLP)](http://veredshwartz.blogspot.jp/2016/08/crowdsourcing-for-nlp.html)
  * データを集めるのに欠かせない、クラウドソーシングの活用方法についての記事。クラウドソーシングに向いているタスク、信頼性担保の方法、料金についてなど実践的な内容が紹介されている。
* [Natural Language Annotation for Machine Learning](http://shop.oreilly.com/product/0636920020578.do)
* [バッドデータハンドブック ―データにまつわる問題への19の処方箋](https://www.amazon.co.jp/dp/4873116406)
* [ガラポン](http://garapon.tv/developer/)
  * APIでテレビの字幕のデータを取ることができるらしい
