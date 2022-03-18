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
* [iMaterialist Challenge on fashion](https://github.com/visipedia/imat_fashion_comp)
  * 100万点をこえるファッション画像のデータセット。8グループ228のラベルがアノテーションされている。
* [rois-codh/kmnist](https://github.com/rois-codh/kmnist)
  * 日本語の崩し文字のMNIST(Kuzushiji-MNIST)。
  * ひらがな全てを含むKuzushiji-49、漢字のデータセットであるKuzushiji-Kanjiも併せて公開されている。
  * 論文中では、単純に認識だけでなくVAEを使った現代字への復元といった使い方も紹介されている。
* [Open Images dataset](https://storage.googleapis.com/openimages/web/index.html)
  * Googleが公開した900万、ImageNetをこえる6000カテゴリのアノテーション済みデータ。こちらから利用可能。中身はURL+ラベルになっているので、怖がらずにcloneして大丈夫。
  * リリース後に更新が行われており、2017/11/16にリリースされたV3では370万のバウンディングボックス、970万のラベル付き画像のデータセット(training data)に拡大している。
  * [2017/7/20にbounding boxのデータが追加された](https://research.googleblog.com/2017/07/an-update-to-open-images-now-with.html)。総計約200万で、学習データ中の120万は半自動で付与(人が確認済み)、validationデータ中の80万は人がアノテーションを行なっている。クラス数は600で一部にはラベルもついている。
  * 2018/4/30にv4がリリースし、公式ページも一新された。バウンディングボックスが1540万/600クラス、総計1900万画像に拡大。
* [iNaturalist](https://sites.google.com/view/fgvc5/competitions/inaturalist)
  * CVPR2018のワークショップFGVC5の開催に伴い公開されたデータセット。元データは、[iNaturalist](https://www.inaturalist.org/)という観察した生き物の写真を撮って記録するアプリから提供されている。
  * 現実に近い状況での画像分類を行うことを目的としており、様々な状況での撮影、似たような種別、カテゴリ間のデータ数の偏りなどが特徴。
  * ラベル数は8,000、総画像数は45万。
* [HDR+ Burst Photography Dataset](http://hdrplusdata.org/dataset.html)
  * HDR+の写真のデータセット。写真の品質を上げるためのフレーム単位の画像(bursts)とそのマージ結果、最終的な処理結果の3つの種類の画像が包含される。
  * これらの写真は実際のAndroidの機種に搭載されているカメラで撮影したもので、現実に近いデータになっている。
  * データセット全体では3,640 bursts(28,461画像)で765GBもあるので、最初はサブセット(153 bursts, それでも37GBあるが)で試すことが推奨されている。
* [Google-Landmarks: A New Dataset and Challenge for Landmark Recognition](https://research.googleblog.com/2018/03/google-landmarks-new-dataset-and.html)
  * 世界各国のランドマークのデータセット。ランドマーク数は3万、写真総数は200万という規模。
  * 画像の局所特徴量を抽出する[DELF](https://github.com/tensorflow/models/tree/master/research/delf)も併せて公開されている。
  * [2019/5/3にv2が公開された](https://ai.googleblog.com/2019/05/announcing-google-landmarks-v2-improved.html)。データ数が500万へと拡張された。
* [Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)
  * 食品の画像データセット。カテゴリ数は101、画像数は101,000と小粋に数字がそろっている。
  * 各食品は250のテストデータと750の学習用データを持っており、双方人が確認している。画像サイズは全て512pxにリスケールされている
* [Columbia University Image Library (COIL-20)](http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php)
  * 物体を一軸回転させて撮影したデータセット。
  * 20種類の物体を5度刻みで時計回りに回転。画像サイズは128x128。
* [Vehicle Make and Model Recognition Dataset (VMMRdb)](https://github.com/faezetta/VMMRdb)
  * 1950年から2016年までに製造・発売された自動車の画像が、メーカ・モデル・年式の3階層で分類されている。画像数は291,752、クラス数9,170。
  * 様々なユーザーが撮影した画像のため、画像サイズや撮影条件にバラツキがある。
  * 3036クラスを使った学習済ResNet-50も公開されている。 
* [TASKONOMY Disentangling Task Transfer Learning](http://taskonomy.stanford.edu/)
  * 全26の画像関連タスクについて、相互の転移しやすさを調べた研究で使用されたデータセット。
  * この検証には当然1画像について全タスク分のアノテーションが必要だが、それが行われている。屋内の画像で、約400万画像が提供されている。
* [The Art Institute of Chicago THE COLLECTION](https://www.artic.edu/collection)
  * シカゴ美術館のコレクションをダウンロードできるサイト。52,000枚の歴史的なアート作品をダウンロードできる。
* [Tencent ML-Images](https://github.com/Tencent/tencent-ml-images)
  * Tencentが大規模な画像データセットを公開。画像数は1700万、カテゴリ数は11000ほど。
  * ImageNetとOpen Imagesの画像から構成され、カテゴリはWordNetをベースに階層構造が付与されている。
* [Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples](http://metalearning.ml/2018/papers/metalearn2018_paper42.pdf)
  * Meta-Dataset: メタラーニングの性能を測るために、様々なデータセットを組み合わせたデータセット。
  * ImageNet/Omniglotなど計10種のデータセットがミックスされている。
* [Large-scale Fashion (DeepFashion) Database](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
  * ファッションの画像を集めたデータセット。画像数は80万、カテゴリ数は50で、これに加え1000の属性(同じ服でも、素材が皮かツイードかなど)が付与されている。
  * これに加え、衣服のキーポイントをかなり細かくアノテーションした[DeepFashion2](https://github.com/switchablenorms/DeepFashion2)がリリースされている。
* [Brno Mobile OCR Dataset](https://pero.fit.vutbr.cz/brno_mobile_ocr_dataset)
  * OCR in the Wildといった趣のデータセット。2113の科学論文のページに対し、様々な人が様々なデバイス(23種)で撮影した画像とテキストが収録されている。
  * 画像数19728、対応する行テキストが50万と大規模なデータセット。
* [FaceForensics](https://github.com/ondyari/FaceForensics/)
  * 画像生成技術の発展で懸念されるFake画像を検知するためのデータセット。
  * 977のYouTube動画をオリジナルとする1000の動画データセットとなっている(ただし顔部分のみ)。生成手法はDeepfakes/Face2Face/FaceSwap /NeuralTexturesの4つ。
* [HJDataset : A Large Dataset of Historical Japanese Documents with Complex Layouts](https://dell-research-harvard.github.io/HJDataset/)
  * 日本の歴史文書からテキストの抽出を行うためのデータセット。縦書きや段組みなど特殊なレイアウトがあり、チャレンジングなデータセットになっている。

## Video

* [YouTube-8M](https://research.google.com/youtube8m/)
  * Googleが公開した800万にも上る動画のデータセット。コンピューター資源が少ない環境でも活用できるように、動画フレームからInception-V3で特徴抽出済みのデータも公開。これがあればTensorFlow(GPU(single))で1日で学習できるらしい。
  * 時間セグメントごとのラベルがアノテーションされた、[YouTube-8M Segments](https://research.google.com/youtube8m/download.html)もリリースされた。
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
* [STAIR Actions](https://stair-lab-cit.github.io/STAIR-actions-web/)
  * 100種類の人の日常動作を収録したデータセット。各カテゴリは、900~1200の動画を持ち、長さは5~6秒ほど。
  * 動画はYouTubeから取得されている。
* [20BN-JESTER/ 20BN-SOMETHING-SOMETHING](https://www.twentybn.com/datasets)
  * 20BN-JESTERはハンドジェスチャーの、20BN-SOMETHINGは日常のデバイス操作(コンセント入れたり冷蔵庫のドア閉めたりなど)のデータセットが公開。それぞれ15万、10万と計25万件というボリューム。
* [Moments in Time Dataset](http://moments.csail.mit.edu/)
  * 3秒の動画に、その動画で何をしているのか(action)のラベルを付与したデータセット。データ数は、100万ほど
  * 人間だけでなく、動物や自然の動きについてもアノテーションされている
* [TRECVID datasets](http://www-nlpir.nist.gov/projects/trecvid/past.data.table.html)
  * TRECの動画を対象とした情報検索(Video Retrieval)の評価用データセット。
  * 動画中のショット単位(Shot Boundary Detection)、シーン単位(Semantic Indexing)の認識や物体追跡(Instance Search)のためのデータセットが公開されている。
* [UCF101](http://crcv.ucf.edu/data/UCF101.php)
  * 人の動作を検出するためのデータセット。101のクラスを持ち、13,320クリップ、27時間分のデータがある。
  * YouTube のデータを元にしている
* [Playing for Benchmarks](http://playing-for-benchmarks.org/)
  * Intel Vision Labが公開している高画質のゲーム動画に対してCVの各種手法のベンチマークを測定できるデータセット
  * 総数25万枚、全フレームに対してGTがアノテーションされており、 semantic segmentation、semantic instance segmentation、3D scene layout、visual odometry、optical flowのベンチマークが測定可能
* [SALSA Dataset](http://tev.fbk.eu/salsa)
  * ACMMM15で公開された人物間の会話を解析するためのデータセット。18人のスタンディングディスカッションの動画、近接センサー,etc.のマルチモーダルなデータが提供されている。
* [BDD100K: A Large-scale Diverse Driving Video Database](http://bair.berkeley.edu/blog/2018/05/30/bdd/)
  * 運転中の画像を収録したデータセット。名前の通り10万の動画から構成される。
  * 各動画は40秒程度・アメリカの各地で撮影されており、様々な天候や時間帯なども加味されている。
  * 動画中のキーフレームにはセグメンテーション/バウンディングボックスなどのアノテーションが施されており、総画像数は1億を超える。

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
* [Apollo Scape](http://apolloscape.auto/scene.html)
  * Baiduが公開した自動運転車用のデータセット。なお、Baiduは自動運転車用プラットフォームである[Apollo](http://apollo.auto/)も公開している。
  * RGBの映像とそれに対するセグメンテーション、また3Dのレベルでも映像・セグメンテーションが提供されている。
  * 画像数は20万でセグメンテーションクラス数は車や自転車、人など25種類。
* [Mapillary Vistas Dataset](https://www.mapillary.com/dataset/vistas)
  * 自動運転のための、路上画像のデータセット。25,000のアノテーション済みの高解像度データを提供。現在も増え続けており、しかも世界各国の画像が含まれている(日本の道路の画像もある)。
  * 2018/7/30に、道路標識などについてより細かく定義した30クラスが追加された(ただ「標識」ではなくスピード制限なのか方向指示なのかなど)。また、アノテーションの精緻化も行われている。
* [SUN database](http://groups.csail.mit.edu/vision/SUN/)
  * 物体認識・シーン認識のタスクにおいてベースラインとなるようなデータセットを提供している。
  * SUN397: 397のシーンでタグ付けされたデータセット
  * SUN2012: (おそらく2012の)タグ付けがされたデータセット
* [PASCAL-Context Dataset](https://www.cs.stanford.edu/~roozbeh/pascal-context/)
  * 物体検出のデータセットである[PASCAL VOC 2010](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/)にセグメンテーションのアノテーションを行ったもの。
  * 400以上のオブジェクトが対象で、学習用に10,103、テスト用に9637枚の画像が用意されている。
* [DAVIS: Densely Annotated VIdeo Segmentation](http://davischallenge.org/)
  * 動画中の物体検知を行うためのデータセット。ビデオシーケンス中の各フレームについて、ピクセル単位のオブジェクトマスクがアノテートされている。
  * DAVIS 2016, 2017と規模を増やしており、DAVIS 2017では学習/検証/テストデータセットでそれぞれ60, 30, 30のシーケンス(シーケンス中のフレーム数は平均69.8、物体数の平均は2.56)が用意されている。
* [Youtube-Objects dataset](https://data.vision.ee.ethz.ch/cvl/youtube-objects/)
  * 10のオブジェクトクラスの名前でYouTubeの動画を検索し、それに対しオブジェクトのバウンディングボックス、また動作領域(Tubes)をアノテートしたデータセット。
  * 1オブジェクト当たりの動画数は9~24本で、動画の時間は30秒~3分とばらつきがある。
* [Team MC^2 : ARC RGB-D Dataset 2017](http://mprg.jp/research/arc_dataset_2017_j)
  * 棚の中におさめられたオブジェクトをロボットに認識、ピッキングさせるAmazon Robotics Challengeのために作られたデータセット。
  * 画像はRGB画像と深度画像の2種類。これに加え、3Dモデルも提供されている
  * アノテーションとして、アイテムごとに色付けしたセグメンテーション画像と、アイテムの四隅の位置(バウンディングボックス)を記録したテキストファイルが提供されている
  * 学習用データは全400シーンx各2回撮影した800枚＋アイテム一つのみ棚に入れた画像(アイテム数40x各10回撮影で400枚、だが公式サイトでは410となっているので何も入っていない棚の画像がある可能性あり)
  * テスト用データは棚にアイテムを入れた全100シーンx各2回撮影の計200枚
* [Matterport3D: Learning from RGB-D Data in Indoor Environments](https://niessner.github.io/Matterport/)
  * 大規模なシーン認識のためのデータセット。90の建物から抽出された10800のパノラマビューに対してオブジェクト/セグメントがアノテーションされている。画像(RGB-D)数は約20万。
  * データセットを扱うためのコードも公開されている([niessner/Matterport](https://github.com/niessner/Matterport))。ここには、ベンチマークも含まれる。
* [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
  * 画像の認識において、単に映っているものを認識するだけでなく、質感などの情報も重要なはず、ということで様々なテクスチャを収集しタグ付したデータセット
  * 全5640画像で、47のカテゴリがアノテーションされている
* [3D_Street_View](https://github.com/amir32002/3D_Street_View)
  * Googleのストリートビューから作成したデータセット。同じ地点をカメラ位置を変えて複数回撮影した画像が収録されており、カメラ位置推定や特徴点一致などのタスクに利用できる(118セットで、総画像数は約2500万)。
  * また、都市全体の3Dモデルも提供されている。
* [The German Traffic Sign Detection Benchmark](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset)
  * 道路標識を検知するモデルを学習させるためのデータセット。
  * 認識する道路標識のサイズは16x16から128x128までと幅広く、また様々な光の状態(朝昼夜など)が加味されている。
  * 学習用に900枚、1.6Gというボリューム。またオンラインでのモデル評価システムも完備されている。
* [Road Damage Dataset](https://github.com/sekilab/RoadDamageDetector/)
  * 道路の損傷(ひび割れや白線の掠れなど)の検知を学習するためのデータセット。
  * 9,053の画像が収録されており、総計15,435の損傷がバウンディングボックスでアノテートされている。また、損傷の分類も行われている。
  * なお、データは日本発で7つの自治体と掛け合い作成したとのこと。
  * [Road Damage Detection Using Deep Neural Networks with Images Captured Through a Smartphone](https://arxiv.org/abs/1801.09454)
* [SUNCG: A Large 3D Model Repository for Indoor Scenes](https://sscnet.cs.princeton.edu/)
  * 実物に近いよう家具などがレイアウトされた、屋内のモデルのデータセット。45,000の異なるシーンが収録されている。
  * シミュレーター?画像だけでなく、実物に近い形にレンダリングした画像も提供されている。また、すべての画像にオブジェクトのラベルが付与されている。
* [InteriorNet](https://interiornet.org/)
  * 2000万にも上る屋内家具レイアウトのデータセット。家具の3Dモデリング、そのレイアウトはプロが担当しており、配置した家具の画像を様々な光源から撮影。
  * インタラクティブなシミュレーション環境も提供している。
  * AlibabaのパートナーであるKujialeという、VRで家具配置を行う会社から提供を受けているよう。ライセンスはGPLv3
* [DeepDrive](http://bdd-data.berkeley.edu/)
  * Berkleyの公開した自動運転車のためのデータセット。様々な天候や都市、時間帯を収録した10万のHD動画(計1100時間!)から構成される。
  * 画像フレーム10万についてはバウンディングボックス(人、バスなど計10クラス)、運転可能領域、白線(走行可能なレーンや横断歩道)がアノテーションされている。
  * 1万についてはピクセルレベルのセグメンテーションがアノテーションされている。
* [HowTo100M](https://www.di.ens.fr/willow/research/howto100m/)
  * キャプション付きの動画データセット。特定の意図を持って行う様々な動作(料理や工作など)について、撮影者が解説している動画を集めている。
  * 動画本数は120万、行動数は23,000にのぼる。
* [Lyft Level 5 AV Dataset 2019](https://level5.lyft.com/dataset/)
  * 自動運転のためのデータセット。
  * Ford FusionにLiDAR(3台)/カメラ(7台)を搭載し記録したデータを基にしており、レーンや横断歩道、信号、駐車位置など細かなセグメントがアノテーションされている。
* [LVIS: A Dataset for Large Vocabulary Instance Segmentation](https://arxiv.org/abs/1908.03195)
  * 16万4千点の画像に対して1200カテゴリ・200万以上のセグメント情報を付与したデータセット。
  * まれにしか出現しないオブジェクトもセグメント情報を持ち、ロングテールなデータセットとなっている。
* [Waymo Open Dataset](https://waymo.com/open/)
  * [Waymo](https://waymo.com) が公開した自動運転のためのデータセット。
  * LiDAR(5台)/カメラ(5台)を使い様々な天候や都市、時間帯を収録した20秒の動画(10Hz)に、合計120万個の2Dラベルと1200万個の3Dラベルがアノテーションされている。
  * [Google Colablatory で実行できるチュートリアル](https://colab.sandbox.google.com/github/waymo-research/waymo-open-dataset/blob/r1.0/tutorial/tutorial.ipynb) が用意されている。
* [PFN Visuo-Tactile Dataset (PFN-VT)](https://github.com/pfnet-research/Deep_visuo-tactile_learning_ICRA2019)
  * 画像と触覚センサーの値をペアにして収録したデータセット。物体の硬さなどを学習するために使える。
* [Pandaset](https://scale.com/open-datasets/pandaset#overview)
  * LiDARセンサーメーカーのHesaiとアノテーションプラットフォームのScaleAIが共同で構築した自動運転車のためのデータセット。
  * 都市部での運転を意識しており、急勾配の坂や歩道、様々な時間帯で収録を行っている(場所はシリコンバレー)。
* [MSeg](https://github.com/mseg-dataset/mseg-api)
  * 複数のSemantic Segmentationのデータセットをマージしたデータセット。
  * 単に合わせるとアノテーションのポリシーやラベルの差異が発生するため、それらを合わせ80,000画像の220,000マスクを調整している。

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
* [MIRO](https://github.com/kanezaki/MIRO)
  * オブジェクトを様々な角度から撮影したデータセット。
  * カテゴリ数は12で、1カテゴリ10種類の物体が含まれる(コップなら、様々なコップが10種類含まれる)。各物体は、様々なアングルから撮影された160の画像を持つ。
  * [RotationNet: Joint Object Categorization and Pose Estimation Using Multiviews from Unsupervised Viewpoints](https://kanezaki.github.io/rotationnet/)
* [Disney Animation Dataset](https://www.disneyanimation.com/technology/datasets)
  * ディズニーが公開しているアニメーションのデータセット
  * 雲のデータセットと、「モアナと伝説の海」の舞台となるモトヌイ島(架空の島)のレンダリングに必要なデータが公開されている。

## Satellite

* [SpaceNet](https://github.com/SpaceNetChallenge/utilities/tree/master/content/download_instructions)
  * 衛星写真のデータセット。建物の領域などがラベル付けされている
  * データを提供しているリポジトリで、可視化のためのツールなども提供されている
  * [SpaceNet7](https://medium.com/the-downlinq/the-spacenet-7-multi-temporal-urban-development-challenge-dataset-release-9e6e5f65c8d5): Planet社の画像を使用したことで解像度が4m単位に高まり、1月ごと2年間の時系列遷移も収録している。建物のfootprintも取られているため、時系列での追跡が可能。]
* [ABCD (AIST Building Change Detection) dataset](https://github.com/faiton713/ABCDdataset)
  * 津波の被害を受けた建物について、無事だったもの(surviving)と洗い流されたもの(washed-away)それぞれについて津波前後の衛星画像を収めたデータセット。
  * 画像は東日本大震災での被害が対象となっており、建物のダメージの判定は震災後の国土交通省の調査に基づいている。
* [Dublin LiDAR dataset](https://geo.nyu.edu/catalog?f%5Bdct_isPartOf_sm%5D%5B%5D=2015+Dublin+LiDAR)
  * ダブリンの上空からLiDARセンサーで取得した点群のデータセット。300点/m2の密度で、上空以外にも垂直面の情報も提供されているので、3Dモデルを作ることも可能。
  * ダウンロードは範囲ごとになっており、各範囲のページに遷移すると右側の「Tools」の中に「All Downloads」が表示されているので、そこからダウンロードできる。
* [Cars Overhead With Context (COWC) dataset](https://gdo152.llnl.gov/cowc/)
  * 車を映した衛星写真のデータセット。解像度は15cm=1pixel単位で、カナダ・ニュージーランド・ドイツなど各国の画像が収録されている。
  * 32,716の車がアノテーションされており、58,247の車ではない物体を含む。
* [Tellus Data Catalog](https://www.tellusxdp.com/ja/dev/data)
  * 衛星データプラットフォームであるTellusで公開されているデータセットの一覧。
  * 衛星画像はもちろん、標高、地表面温度、降雨量、また携帯電話の基地局アクセスを元にした人口統計情報など興味深いデータが多く公開されている。
* [AGRICULTURE-VISION DATABASE](https://www.agriculture-vision.com/dataset)
  * 農地の衛星画像データセット。雲の影や農地境界の種別?、水路などといった農業にとって重要なラベルのアノテーションがされている。
* [The GeoLifeCLEF 2020 Dataset](https://github.com/maximiliense/GLC)
  * どのような生物種がどこに生息しているのかを、衛星画像と共に収録したデータセット。iNaturalistという発見した動植物を位置情報付きで投稿できるアプリのデータをベースに作成している。ただカバーされているのは今のところアメリカとフランスのみ。

## BodyParts

* [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  * 20万件の顔画像データと、それに撮影場所や40の特性(笑っているとか、ヒゲがあるとか)がセットになったデータ。また一万件程度は身元のデータがある(有名人などの画像)。
* [MegaFace and MF2: Million-Scale Face Recognition](http://megaface.cs.washington.edu/)
  * 約500万の画像からなる顔のデータセット。70万人分あり、一人当たり平均7画像が用意されている(最小3、最大2500近くとばらつきは結構大きい)
  * 顔を囲ったBounding Boxのデータも併せて提供されている。
* [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
  * 約300万の画像からなる顔のデータセット。9131人分あり、一人当たり平均362.6画像が用意されている。
  * 画像はGoogle Image Searchから取得されたもので、ポーズや年齢、人種など多様な人の画像が含まれる。
* [IMDb-Face](https://github.com/fwang91/IMDb-Face)
  * IMDbに掲載されている映画のスクリーンショットやポスターから抽出した画像に、人手でアノテーションしたデータセット。
  * 既存のデータセット(MegaFace/MS-Celeb-1Mなど)はラベルのノイズが多く、実際は20~30%の量で同等のパフォーマンスが出せる、とした研究で作成された([The Devil of Face Recognition is in the Noise](https://arxiv.org/abs/1807.11649))。
* [DiF: Diversity in Faces Dataset](https://www.research.ibm.com/artificial-intelligence/trusted-ai/diversity-in-faces/)
  * 件数100万件という、大規模な人の顔のデータセット。
  * 顔だけでなく、顔の特徴のアノテーションも行われている(顔の長さ、鼻の長さなど)。
* [Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset)
  * GANでの使用を想定した顔のデータセット。1024x1024という高解像度の画像が70,000枚収録されている。
  * 画像は名前の通りFlickrから取得されており、メガネや帽子など様々なバリエーションが用意されている。
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
  * データセットの信頼性について疑義あり。放射線技師の方が実際に画像を見て検証したところ、診断と異なるラベルが多く発見された(ものによると20~30%台の一致しかないものもある)。詳細はこちら=>[Exploring the ChestXray14 dataset: problems](https://lukeoakdenrayner.wordpress.com/2017/12/18/the-chestxray14-dataset-problems/)
* [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)
  * 胸部X線画像のデータセット。65,240人の患者から撮影した、224,316の画像が収録されている。
  * 放射線技師の所見(テキスト)を解析し、14の病理について有無のラベルを付けている。
* [標準ディジタル画像データベース　胸部腫瘤陰影像(DICOM版)](http://imgcom.jsrt.or.jp/download/)
  * 154の小瘤のある画像と、93のない画像で構成されるデータセット。解像度は2048 x 2048。
  * 北里大学メディカルセンター放射線部 柳田 智先生のご尽力により作成された
* [MIMIC](https://mimic.physionet.org/)
  * 40,000人のケアが必要な重篤な患者についてのデータセット。人口統計、バイタルサイン、検査結果、医薬品情報などが含まれる。
  * 利用にあたってはまずCITIの"Data or Specimens Only Research"というオンライン講座を受講する必要がある([こちら](https://mimic.physionet.org/gettingstarted/access/)参照)。
* [DeepLesion](https://www.nih.gov/news-events/news-releases/nih-clinical-center-releases-dataset-32000-ct-images)
  * 32,000枚のCTスキャン画像のデータセット。画像データセットとしてはかなり大規模。
  * 匿名化された4,400名の患者の画像で、CTスキャンの後にチェックすべき箇所(病変)を画像に書き込むらしいのだが、そのチェックが付与されているとのこと。
* [fastMRI](https://fastmri.med.nyu.edu/)
  * 膝のMRI画像で、10,000の画像からサンプルされた1500件のデータが提供されている。
  * フォーマットは、ベンダー中立のISMRMDが使われている。
* [MRNet Dataset](https://stanfordmlgroup.github.io/competitions/mrnet/)
  * 膝のMRI画像データセット。データ数は1,370で、前十字靭帯/半月板の損傷についてアノテーションが行われている。
* [MURA](https://stanfordmlgroup.github.io/competitions/mura/)
  * 骨のX線画像のデータセット。
  * 画像数は40,561で、肘、指など7つの部分に分かれる。正常/異常の判定がアノテーションされている。
* [Optos images dataset](https://tsukazaki-ai.github.io/optos_dataset/)
  * 姫路にあるツカザキ病院から、眼底画像のデータセットが公開。
  * 5389名の患者から撮影した13,000ほどの画像が収録されており、患者の性別、右目or左目、疾患(全9種類)がラベル付けされている。
* [COVID-19 Image Data Collection](https://github.com/ieee8023/covid-chestxray-dataset)
  * 新型コロナウィルス患者の胸部X線画像(公開時点では正面画像123枚)。
  * データは公開論文や放射線画像リファレンスサイト([Radiopaedia](https://radiopaedia.org/))から取得されている。重複削除等についての言及はないが、現在進行形でブラッシュアップされているよう。
* [4CE Consortium for Clinical Characterization of COVID-19 by EHR](https://covidclinical.net/index.html)
  * i2b2とOMOPの呼びかけで、世界各国の病院(アメリカ、イタリア、ドイツ、シンガポール、フランス、計96の病院)から新型コロナウィルスの電子カルテデータを集めて統合。計27,927件で検査値の総量は187,802件に上る。医療機関(国?)によりやはり違いがあるよう。
* [DELVE Global COVID-19 Dataset](https://rs-delve.github.io/data_software/global-dataset.html)
  * 様々な新型コロナウィルスに関するデータをまとめたデータセット。各国の対策、検査/死亡率、また気候データなど国横断で差異を意識しながら分析できるようまとめらている。pandasですぐに読み込めるようになっており、扱いやすい。
* [CLIP: A Dataset for Extracting Action Items for Physicians from Hospital Discharge Notes](https://physionet.org/content/mimic-iii-clinical-action/1.0.0/)
  * 退院記録をアノテーションしたデータセット。専門医療センターで測定した患者のバイタルサイン、投薬、検査結果等のデータセットMIMIC-IIIに対し行っている。退院後の患者のケアへの活用を企図している。

## Art

* [Painter by Numbers(PBN)](https://www.kaggle.com/c/painter-by-numbers/data)
  * 画家・タイトル・画風・ジャンルなどがアノテーションされた画像のデータセット
  * 全23817画像あるが、かなり重たいので(学習用データセットが36GB)アノテーションの配分が均等になるように分けられた13グループのファイル(各1400~2000画像くらい)が用意されている
* [quickdraw-dataset](https://github.com/googlecreativelab/quickdraw-dataset)
  * 345カテゴリ、5千万のイラスト描画データ。描画データという名の通り、時系列の筆跡データが提供されている
  * 追加のための学習データセットとして、漢字や羊、象形文字?のデータも公開されている。[sketch-rnn-datasets](https://github.com/hardmaru/sketch-rnn-datasets)
  * 本体のモデルは[こちら](https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn)。
  * [A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477)
* [How Do Humans Sketch Objects? (TU-Berlin dataset)](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/)
  * 人の書いたスケッチのデータセット。20,000のイラストに対しカテゴリが付与されており、カテゴリ数は250上る。
  * 人間がカテゴリを判別できる精度は73%(SVMだと56%)。
* [SketchyScene](https://github.com/SketchyScene/SketchyScene)
  * 7000超のシーンのテンプレート/写真に沿った、29000ほどのイラストが収録されている。さらに、セマンティック/インスタンスレベルのセグメンテーションデータが付属している。
* [Manga109](http://www.manga109.org/ja/)
  * 1970年代から2010年代に、実際に出版された109冊の漫画のデータセット。一部の巻についてはページ毎のセリフデータも用意されている。
  * 利用は学術目的限りで、論文などにデータセットの漫画を掲載する際は作者の著作権「©作者名」とManga109の利用であることを明示すること(詳細は上記サイト参照)。
  * 有志がアノテーションデータを読み込むためのスクリプトを開発: [matsui528/manga109api](https://github.com/matsui528/manga109api)
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
* [AADB dataset](https://github.com/aimerykong/deepImageAestheticsAnalysis)
  * 写真の審美性について1~5のスコアがつけられたデータセット。10,000の画像が含まれる。
  * AVA(Image Aesthetic Visual Analysis) datasetよりも画像数は少ないが(255,000)、0/1でなく1~5でスコアがついている点が強味
  * [Photo Aesthetics Ranking Network with Attributes and Content Adaptation](https://arxiv.org/abs/1606.01621)
* [Cartoon Set](https://google.github.io/cartoonset/index.html)
  * 二次元のアバターイメージのデータセット。顔は色違いやパーツの組み合わせ違いのパターンで作られており、各特性のラベルがついた1万/10万の2つのデータセットが提供されている。
* [KaoKore Dataset](https://github.com/rois-codh/kaokore)
  * 日本画の中にある顔画像のデータセット。主に室町時代末～近世初期に作られた絵本・絵巻物から顔画像を抽出しメタデータをアノテーションしたデータセット([顔貌コレクション](http://codh.rois.ac.jp/face/))から5552画像を抽出し、256x256にリサイズし性別/社会ステータスなどのラベルを振っている。

## Image Captioning/Visual QA

* [VQA](http://www.visualqa.org/index.html)
  * 画像を見て質問に答えるタスクでは、学習した画像についてだけ答えられる、良くある答え(「2つ」とか)を多めに繰り出して精度が上がっているなど明らかな過適合が見られた。そこで真実見たことない画像(Zero-Shot)に回答可能かをテストするためのデータとベースラインモデルの提案
  * MS COCOから204,721画像、50,000のクリップアートが含まれる。各画像には最低3つ質問があり、各質問には10の回答がある
  * 回答の種類にはバリエーションがあり、アノテーターが最も多く回答したもの(Correct)、画像を見ないで回答したもの(Plausible)、よくある回答(Popular: 2つ、とかyes、など)、回答からランダムに選択したもの、などがある。
  * [Zero-Shot Visual Question Answering](https://arxiv.org/abs/1611.05546)
* [VizWiz Dataset](http://vizwiz.org/data/#dataset)
  * 画像を見て質問に答えるVQAを、盲目の人が周りを認識するという実用的なシーンで役立てるために作成されたデータセット。
  * 画像、また質問は実際に盲目の人が(モバイル端末で)撮影、また質問したもので、それに対する回答をクラウドソーシングで収集している。
  * 31,000の質問と、各質問に対する回答が10収録されている。
* [CLEVR](http://cs.stanford.edu/people/jcjohns/clevr/)
  * 画像理解のためのデータセット。きちんと理解しているかを診断するために、シンプルな画像(物体がいくつか置いてあるような画像)に対し、様々な内容(物体の色や形といった属性、個数、位置など)を問う質問が用意されている
  * 質問は自然言語の形式だけでなく、プログラムの表現に近い形での形式も用意されている(関数をつなげているような形)。
* [MS COCO](http://mscoco.org/home/)
  * 認識・セグメンテーション等のデータを含む物体認識のための統合的なデータセット
  * 画像に対する5つの説明(キャプション)も含む
* [COCO-Stuff 10K](https://github.com/nightrome/cocostuff)
  * COCOのデータセット(の一部)にピクセルレベルのアノテーションを行ったデータセットが公開。10,000の画像に91の物体(人や船、象など)がピクセル単位でアノテーションされている。
  * その後164,000に拡張され、インスタンスレベルでのアノテーション(同じクラスでも別々の個体なら区別する)も追加されている。
* [VisDial Dataset](https://visualdialog.org/data)
  * MS COCOの画像をベースに、それについてのQとAが付与されている。Training用に8万、Validation用に4万が利用可能
  * [アノテーションツールまで公開されている](https://github.com/batra-mlp-lab/visdial-amt-chat)ので、さらに集めることも可能。
* [Toronto COCO-QA Dataset](http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/)
  * MS COCOの123,287画像に対し、学習用に78,736、評価用に38,948と大量のQAを作成したデータセット。
  * これにはカラクリがあり、QAはMS COCOのキャプションから自動生成されている(「椅子が二つある」というキャプションなら、Q:何個椅子があるか? A:2つ、など)。そのため、質問の種類は限定的で文法的におかしい質問も含まれる。
* [DAQUAR - DAtaset for QUestion Answering on Real-world images](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/vision-and-language/visual-turing-challenge/)
  * [NYU Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)の画像について作成されたQAのデータセット。学習用に6794、評価用に5674の質問が用意されている(1画像当たり平均9個)。
  * ただ、画像が屋内のもので角度/光の当たり的に回答がそもそも難しい質問も多くあり、人間の正答率も50.2%と低い。
* [STAIR Captions](https://stair-lab-cit.github.io/STAIR-captions-web/)
  * MS COCOの画像につけられた5つのキャプションについて、日本語でもキャプションをつけたもの(翻訳したわけではなく、独自にアノテートしている)。
  * [STAIR Captions: Constructing a Large-Scale Japanese Image Caption Dataset](https://arxiv.org/abs/1705.00823)
* [Cornell NLVR](http://lic.nlp.cornell.edu/nlvr/)
  * 図と、その図の状態を記述した自然言語のデータセット(具体的には、「少なくとも一つの黒い三角形がある」とか)。
  * [A Corpus of Natural Language for Visual Reasoning](http://alanesuhr.com/suhr2017.pdf)
* [Recipe1M dataset](http://im2recipe.csail.mit.edu/)
  * 100万にも及ぶ、料理画像とそのレシピのペアのデータセット(料理の種類としては80万)。
  * なお、利用は研究用途に限る。
* [Food-Recipe-CNN](https://github.com/MURGIO/Food-Recipe-CNN)
  * chefkoch.deというレシピサイトから収集された料理の画像40万、レシピ数30万という大規模なデータセット。カテゴリ数は230。
* [ShapeNet Text Descriptions](http://text2shape.stanford.edu/)
  * ShapeNetの3D画像(椅子・テーブルのみ)について、説明文をペア付与したデータセット(茶色の椅子、など)。
  * 研究では言語からの3D生成も行っており、ボクセルデータも付属している。
* [RecipeQA](https://hucvl.github.io/recipeqa/)
  * レシピに関する対話データセット。レシピを読んで回答するというのが基本スタイルだが、4つの回答パターンが用意されている。
  * Visual Coherence: レシピに一致しない画像の回答、Visual Cloze: 欠けたレシピ手順の画像選択、Visual Ordering: レシピ手順通りに並んだ画像の選択、Textual Cloze: レシピを完成させるための手順(テキスト)の選択、という4タスク。
* [NLVR](https://github.com/clic-lab/nlvr)
  * 画像に関する質問に答えるVQAのデータセット。既存のデータセットは画像認識とほぼ等価なところがあるため、より複雑な質問になるよう工夫されている。
  * 具体的には、単一の画像ではなく2つの画像を使用し、比較するようなキャプションをつけている。データ数は107,296件。
* [MMD: Towards Building Large Scale Multimodal Domain-Aware Conversation Systems](https://amritasaha1812.github.io/MMD/)
  * マルチモーダル対話のためのデータセット。
  * ファッションドメインのデータセットで、「ユニクロの1000円で買えるTシャツが欲しい」「こちらはいかがでしょう(画像を提示)」といったような画像を使用した対話データが収録されている(実際は英語)。
  * 15万対話と数も大規模。
* [Touchdown](https://github.com/clic-lab/touchdown)
  * 自然言語によるナビゲーションのデータセット。道案内だけでなく、たどり着いた場所に隠されたオブジェクト(🐻)を見つけるタスクになっている。
  * データはGoogle Street Viewからとられており、件数は9,326。 
* [VSU: Visual Scenes with Utterances Dataset](https://github.com/yahoojapan/VSU-Dataset)
  * 人の写っている画像に対し視線をアノテーションしたGazeFollowに対して、人物がなんと言っているかを追加アノテーションしたデータセット。
  * クラウドソーシングを使って発話内容を作成しており、発話についてはダブルチェック/人手チェックを行なっている。
  * 同様のデータセットとしては、[VCR: Visual Commonsense Reasoning](http://visualcommonsense.com/)がある
* [GQA](https://cs.stanford.edu/people/dorarad/gqa/)
  * 既存のVQAデータセットは回答に偏りがあり(いくつ?ならtwoが多いなど)根拠に基づく学習をさせるのが難しく、根拠を意識したデータセット(CLEVR)は四角や三角などの抽象的なオブジェクトという問題があった。この2つを合流させたようなデータセットになっている。
* [ActivityNet Captions dataset](https://cs.stanford.edu/people/ranjaykrishna/densevid/)
  * 動画に対するキャプションをつけたデータセット。動画全体で1つではなく、動画内の様々なパートに対しアノテーションが行われている。
  * 動画数は2万で、一動画あたり約3.65パートに文が付与されている。
* [YouCook2 Dataset](http://youcook2.eecs.umich.edu/)
  * 料理動画のデータセット。89のレシピ X 22動画/1レシピ = 計2000本ほどが収録されている。動画はYouTubeから取得され、手順に対しキャプションがつけられている。
* [VFD Dataset](https://github.com/yahoojapan/VFD-Dataset)
  * 日本語の画像つき対話データセット。エージェントの一人称視点画像・人間話者の視線に対し、人間の発話とそれに対するエージェントの言語/非言語の反応がアノテーションされている(アノテーションはすべてテキスト)。
  * 画像は[GazeFollow](http://gazefollow.csail.mit.edu/index.html)のデータが使用されている。
* [open-mantra-dataset](https://github.com/mantra-inc/open-mantra-dataset)
  * 漫画の翻訳を学習するためのデータセット。漫画の画像に加えて、コマの位置・吹き出しの位置(x, y, w, h)・吹き出し内のテキスト(日英中)がアノテーションされている。

# NLP

* [自然言語処理のためのリソース](http://nlp.ist.i.kyoto-u.ac.jp/index.php?NLP%E3%83%AA%E3%82%BD%E3%83%BC%E3%82%B9#g63a7f30)
  * 京都大学から提供されている自然言語処理のためのデータセット。毎日新聞のデータに対する各種言語情報である[京都大学テキストコーパス](http://nlp.ist.i.kyoto-u.ac.jp/index.php?%E4%BA%AC%E9%83%BD%E5%A4%A7%E5%AD%A6%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E3%82%B3%E3%83%BC%E3%83%91%E3%82%B9)(※本文データは別途入手する必要がある)、さまざまなウェブ文書のリード文に対し各種言語情報のアノテーションを行った[京都大学ウェブ文書リードコーパス](http://nlp.ist.i.kyoto-u.ac.jp/index.php?KWDLC)等がある。
* [解析済みブログコーパス](http://nlp.ist.i.kyoto-u.ac.jp/kuntt/)
  * 京都大学と、NTTコミュニケーションの共同研究ユニットで作成されたコーパス。4テーマ（京都観光、携帯電話、スポーツ、グルメ）、249記事、4,186文の解析済みブログコーパス。形態素、構文、格・省略・照応、評判情報がアノテーションされている。
* [Stanford Rare Word (RW) Similarity Dataset](https://nlp.stanford.edu/~lmthang/morphoNLM/)
  * 文中にあまり登場しない低頻度語について、余りレアではない単語とペアにして、その類似度を付けたデータセット
  * 類似度の評価は、クラウドソーシングで0-10の11段階で評価をしてもらい、つけている。
  * [信頼性について疑問符が付くという報告有り](https://medium.com/@taher.pilevar/is-the-stanford-rare-word-similarity-dataset-a-reliable-evaluation-benchmark-3fe409053011)。低頻度語がどんな語に似ているかは一定の知識が要求されるため、クラウドソーシングには向かないのではないかという指摘もある。
* [日本語単語類似度データセット(JapaneseWordSimilarityDataset)](https://github.com/tmu-nlp/JapaneseWordSimilarityDataset)
  * Stanford Rare Word Similarity Datasetを参考に作成された日本語の単語類似度データセット。
  * 動詞・形容詞・名詞・副詞が対象となっており、クラウドソーシングを利用し10名のアノテータに11段階で単語ペアの類似度をスコアしてもらっている。
* [The Japanese Bigger Analogy Test Set (jBATS)](http://vecto.space/projects/jBATS/)
  * 日本語のアナロジータスク(王様-男+女=女王、など)のデータセット。本家のBATSに乗っ取り、4つの言語関係が収録されている。
* [WikiText](https://metamind.io/research/the-wikitext-long-term-dependency-language-modeling-dataset/)
  * 言語モデル学習用のデータセットWikiText-2/WikiText-103の公開。それぞれPenn Treebankよりも2倍&110倍のデータ量。
* [WikiSQL](https://github.com/salesforce/WikiSQL)
  * 自然言語をSQLに対応づけたデータセット。なお、SQLは選択用(SELECT)のみ。
  * 自然言語とSQLの条件・列選択・集計対象列との対応データと、テーブルの定義が提供されている。
* [ParaphraseBench](https://github.com/DataManagementLab/ParaphraseBench)
  * 自然言語とSQLを対応づけたデータセット。結果となるSQLは同じだが、6つの異なる聞き方をした自然言語を収録している。
  * データ件数は57とそれほど多くない。学習用というよりは、評価用のデータセット([論文中](https://arxiv.org/abs/1804.00401)では、こうしたフォーマットを元にデータを水増ししている)。
* [text2sql-data](https://github.com/jkkummerfeld/text2sql-data)
  * これまで公開された7つのText2SQLのデータを統合し、かつ実際の大学生の質問から作成したデータセットを新たに追加。
  * 各データの学習/評価データについては、結果として同じSQLになるものが互いに含まれないようにしている。
* [Spider 1.0](https://yale-lily.github.io/spider)
  * Text to SQLのデータセット。質問数は10,181で、対応付けられるSQL数は5,693。これらのSQLは、138の異なるドメインの200のデータベースに対するアクセスを行うものになる。
* [青空文庫](http://www.aozora.gr.jp/)
  * 著作権の消滅した作品、また「自由に読んでもらってかまわない」とされたものをテキストとXHTML(一部HTML)形式に電子化した上で揃えている
  * [GitHubからダウンロードが可能になった](https://github.com/aozorabunko/aozorabunko)。
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
* [The General Language Understanding Evaluation (GLUE) benchmark](https://gluebenchmark.com/)
  * 自然言語理解を行うモデルの性能測定プラットフォームGLUE(データセットを含む)。
  * 内容としては、質問回答や感情分類、テキスト間の関係推定などのタスクからなり、単一のモデルがこれら複数のタスクをどれだけこなせるかを測定する。
  * [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://arxiv.org/abs/1804.07461)
* [PeerRead](https://github.com/allenai/PeerRead)
  * ACL/NIPS/ICLRの論文の概要と、それに対するレビューを集めたデータセット(accept/rejectされたもの双方が含まれる)。論文数は14,000で、レビュー数は10,000。
* [kanjivg-radical](https://github.com/yagays/kanjivg-radical)
  * 漢字の部首、そして部首と漢字を対応付けたデータセット。
  * 「脳」なら「月」「⺍」「凶」、また「月」という部首からなら「肝」「育」などが取れる。詳細は[記事](https://yag-ays.github.io/project/kanjivg-radical/)を参照。
  * 同様のデータセットとして、[IDS](https://github.com/cjkvi/cjkvi-ids)がある。ただ、こちらはライセンスがGPLv2なので注意。
* [CoNaLa: The Code/Natural Language Challenge](https://conala-corpus.github.io/)
  * 自然言語からコードを生成するためのデータセット。
  * 「ソートしたい」=>「配列xを大きい順に並び変えたい」=>「x.sort(reverse=True)」といった形で、要求・具体的な要求・コード(Python)の3点がセットになっている(クラウドソーシングで作成、データ元はStackOverflow)。
* [MLDoc](https://github.com/facebookresearch/MLDoc)
  * 文書分類のデータセットであるReuters corpusを加工して作成されたデータセット。
  * 各クラスのバランスがとられているほか、英語を含めた8言語への翻訳テキストも含んでいる(日本語も含まれる)。
* [ToTTo](https://github.com/google-research-datasets/totto)
  * テーブルからデータを読み取り文にするデータセット。
  * テーブルの情報と指定セルを入力に、読み取れる結果の文がアノテーションされている(東京都の人口のテーブルで江東区のセルがハイライトされていた場合、「東京江東区の人口はxx人です」が回答文になるなど)。
* [KILT](https://github.com/facebookresearch/KILT)
  * 知識ベースに基づく複数タスクでの性能を測るためのベンチマークデータセット。元データはWikipediaで、QAや対話、Entity Link、ファクトチェックといったタスクが収録されている。
* [MeDAL dataset](https://github.com/BruceWen120/medal)
  * 医療略語の明確化を事前学習として行うためのデータセット。3つ以上の略語を含む1400万文書を含んでおり、これを事前学習に用いることで後続の医療タスクの精度を向上させることができる。
* [GEM](https://gem-benchmark.com/)
  * 自然言語生成のベンチマーク。多言語のものを含む全13データセットで評価が行われる。単純なROUGE評価だけでなく、人手評価も検討されているよう。データロードだけでなく転移学習可能なツールも提供されている。
* [Contract Understanding Atticus Dataset](https://github.com/TheAtticusProject/cuad/)
  * 商取引の契約書にアノテーションを行ったデータセット。500超の契約書文について、組織名などの契約者情報、トラブルになりそうな箇所、コスト負担になりそうな箇所の3種類41ラベルが専門家によりアノテーションされている。
* [Dynabench](https://dynabench.org/)
  * 自然言語処理の性能を評価するために「動く」ベンチマーク。敵対サンプルの考えを使い、モデルが間違えるサンプルを人間が登録していくことでベンチマークを更新していく。サンプル作成は実際サイト上で試すことができる。

## Parallel Corpus

* [日本語対訳データ](http://phontron.com/japanese-translation-data.php?lang=ja)
* [Tanaka Corpus](http://www.edrdg.org/wiki/index.php/Tanaka_Corpus)
  * 日英翻訳のためのパラレルコーパス。約15万文の日英の分のペアが収録されている。
  * こちらから単語数が4~16である文約5万件を抽出した、単語分割済みのコーパスが別途公開されている([small_parallel_enja](https://github.com/odashi/small_parallel_enja))。
* [JESC: Japanese-English Subtitle Corpus](https://nlp.stanford.edu/projects/jesc/)
  * インターネット上から取得した映画・テレビ番組の字幕に対して作成された日英のパラレルコーパス。
  * 320万文が含まれる
  * [JESC: Japanese-English Subtitle Corpus](https://arxiv.org/abs/1710.10639)
* [SNOW T15:やさしい日本語コーパス](http://www.jnlp.org/SNOW/T15)
  * 日英対訳コーパス([small_parallel_enja](https://github.com/odashi/small_parallel_enja))の日本語を平易な日本語に書き換えたデータセット。
  * 元がパラレルコーパスであるため、英語との対応もとれる。
* [TED-Parallel-Corpus](https://github.com/ajinkyakulkarni14/TED-Multilingual-Parallel-Corpus)
  * TED Talkをベースにしたパラレルコーパス。対応言語が非常に多く、109言語のリソースが収録されている。
* [JParaCrawl](http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/)
  * 数多のWebサイトからツールを駆使して対訳文を検出し作成したデータセット。1000万の英日、8万3千の中日文が収録されている。
* [OPUS](http://opus.nlpl.eu/)
  * オープンに使える、様々な言語のパラレルコーパス集。有志によりWeb上のフリーなコンテンツから作成されている。
* [CCMatrix: A billion-scale bitext data set for training translation models](https://github.com/facebookresearch/LASER/tree/master/tasks/CCMatrix)
  * 576言語のパラレルコーパス。文はCommonCrawlから収集されており、文数は45億にもおよぶ。異なる言語の類似度判定には[LASER](https://github.com/facebookresearch/LASER/tree/master/tasks/CCMatrix)が使われているよう。
* [The Business Scene Dialogue corpus](https://github.com/tsuruoka-lab/BSD)
  * ビジネスシーンの対話を収録した日英の対訳コーパスが公開。電話や会議、研修やプレゼンといったシーンを定めそれぞれに対話が作成されている。
* [FLORES-101](https://github.com/facebookresearch/flores?fbclid=IwAR0TO2pdTMqf8LphTgNYPAHtuwHgDpfBP1-V3g6EtAyZwMOYWYGq5lmLeF0)
  * 多言語翻訳モデルのモデルを評価するためのベンチマーク。さまざまなトピックのWikipediaから抽出した3001文をプロの翻訳家が101言語に翻訳した文を収録している。

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
* [The Stanford Natural Language Inference (SNLI) Corpus](https://nlp.stanford.edu/projects/snli/)
  * 二つの文の関係を推定するためのデータセット。論理的に同じ意味か、矛盾するものか、どちらともいえないかの3種類。「男は外でコーヒーを飲んでいる」vs「男は寝ている」の場合、矛盾している、など。
  * 文だけでなく、構文解析した結果も含まれる。そのため、Recursiveなモデルによる意味獲得の評価などによく用いられる。
  * ただし、データセットの作り方に問題があり、[片方の文だけで分類が推定されてしまうという点が指摘されている](https://github.com/arXivTimes/arXivTimes/issues/670)。
* [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/)
  * SNLIを、多様なジャンルの話題に拡張したデータセット。話し言葉や書き言葉といった違いのあるジャンルも含まれる。
* [XNLI: Evaluating Cross-lingual Sentence Representations](https://arxiv.org/abs/1809.05053)
  * MultiNLIを、15カ国語に拡張したデータセット。この中には、ウルドゥー語などの少ない言語資源のデータも含まれる。
  * (論文中にzipファイルの直リンクが書いてある)。
* [e-SNLI](https://github.com/OanaMariaCamburu/e-SNLI)
  * 文の関係を推定するデータセットであるSNLIに、関係の理由をアノテーションしたデータセット。
  * 例:「ステッキを持って歩いている」と「手ぶらで歩いている」は矛盾する=>理由: ステッキを持っているということは手がふさがっているから、など。
* [Enron Email Dataset](https://www.cs.cmu.edu/~enron/)
  * エンロン事件の捜査のさなか、米連邦エネルギー規制委員会(FERC)がインターネット上に公開した実際のエンロン社内のメールデータ。
  * 件数は50万件ほどで、主にエンロンのシニアマネージャーの人達が送ったもの。ユーザー数的には150名ほど。なお添付ファイルは削除されている。
  * メールのデータセットに対して、その意図("要求"か"提案"か)をアノテートしたデータセットが公開されている([EmailIntentDataSet](https://github.com/ParakweetLabs/EmailIntentDataSet))。
* [PubMed 200k RCT dataset](https://github.com/Franck-Dernoncourt/pubmed-rct)
  * 連続する文の分類を行うためのデータセット。具体的には、論文のAbstractに対してこの文は背景、この文は目的、この文は手法・・・といった具合にアノテーションされている。
  * 20万のAbstractに含まれる、230万文にアノテーションが行われている。
  * [PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts](https://arxiv.org/abs/1710.06071)
* [FAKE NEWS CHALLENGE STAGE 1 (FNC-I): STANCE DETECTION](http://www.fakenewschallenge.org/)
  * フェイクニュースの検知を目的としたデータセット。第一弾として、記事のスタンスの検知を行うデータを提供している。
  * Inputは記事のタイトル/本文で、それに対し他の記事を賛成・反対・同じことを話しているがスタンスはとっていない、関係ないの4つに分類する。
* [STS Benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark)
  * 文書類似度のタスクのためのデータセット(SemEval2012~2017で使用されたもの)。画像のキャプションやニュース記事などが含まれる。
* [The SICK data set](http://clic.cimec.unitn.it/composes/sick.html)
  * 1万の英語の文書について、文間の類似性や関係性をアノテーションしたデータセット。
* [MultiFC: A Real-World Multi-Domain Dataset for Evidence-Based Fact Checking of Claims](https://copenlu.github.io/publication/2019_emnlp_augenstein/)
  * ファクトチェックを行うためのデータセット。
  * 26のファクトチェックサイトのデータを収集して作成されており、合計36,534のチェック結果が収録されている。
  * ただ、ラベル数がサイトによって異なる(AサイトのデータのラベルとBサイトのラベルは異なる)。

## Sentiment

* [Sentiment Treebank](https://nlp.stanford.edu/sentiment/code.html)
  * Stanfordの公開している、意味表現ツリーのデータセット
* [Sentiment140 - A Twitter Sentiment Analysis Tool](http://help.sentiment140.com/for-students/)
  * Tweetにネガポジのラベルを付与したデータセット。データ数は160万件で、ポジティブ80万・ネガティブ80万できちんとバランスしている。
  * なお、Tweet関係のデータは[こちらの論文](https://arxiv.org/abs/1601.06971)によくまとまっている。
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
* [Twitter日本語評判分析データセット](http://bigdata.naist.jp/~ysuzuki/data/twitter/)
  * 主に携帯電話に関するツイートを集め、センチメントのアノテーションを行ったデータセット。件数は534,962件と、かなり大規模。
  * 携帯電話の機種/会社に関するざっくりとしたカテゴリのラベルが付与されており、センチメントはその対象について「言及されていた場合にのみ」アノテーションされているので注意。
  * また、データセットにはTwitterのIDのみ含まれ、本体のテキストは含まれていない。
* [SNOW D18:日本語感情表現辞書](http://www.jnlp.org/SNOW/D18)
  * 日本語の感情表現を集めた辞書。2000の表現が48の感情に分類されている。
  * アノテーターは3名で、アノテーターごとの結果を利用できる(集約は行われていない)

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
* [Broad Twitter Corpus](https://gate.ac.uk/wiki/broad-twitter-corpus.html)
  * Twitterのつぶやきに対して固有表現をアノテートしたデータセット。
  * 多様な地域や年代、内容をカバーしているのが特徴であり、地域は英語圏だが年代は2012~2014年、内容は災害やニュース、フォロワーの多い著名人のものなど多岐にわたっている。
  * アノテーション数は総計12,000。
* [litbank](https://github.com/dbamman/litbank)
  * 文学に特化した固有表現認識のデータセット。ストーリーを追うために、人物・場所/建物・移動手段、また組織といった固有表現についてアノテーションが行われている。対象は100作品で、各作品から単語数にして2000単語ほどが抽出されアノテーションされている。
  * アノテーション数は総計210,532。

## Knowledge Base

* [Visual Genome](http://visualgenome.org/)
  * 画像とその物体の名前、それらの関係性、またQAなどを含む認識理解に必要なデータを包括しているデータセット
* [Microsoft Concept Graph](https://concept.research.microsoft.com/Home/Introduction)
  * Microfostが公開した、エンティティ間の関係をについてのデータセット。最初はIsA関係(AはBだ的な)のデータで、1,200万のインスタンスと、500万のコンセプト間の、8500万(!)のisA関係を含んでいる。
* [mmkb](https://github.com/nle-ml/mmkb)
  * 知識グラフとしてよく利用されるFB15k、YAGO15k、DBpedia15kについて、その数値データ(緯度経度など)を付与したもの。
  * 画像データについても付与が行われている。
* [The TAC Relation Extraction Dataset](https://nlp.stanford.edu/projects/tacred/)
  * 大規模な関係認識のデータセット。アノテーションはNewswireなどのニュース記事やその他Web上の記事に対して行われている。
  * 主語・述語をはじめとした様々な関係が付与されている。データ総数は約10万。
  * ただし有料(LDC memberは無料、それ以外は$25)。
* [S2ORC: The Semantic Scholar Open Research Corpus](https://github.com/allenai/s2orc/)
  * Semantic Scholarのデータを基にした、8100万の論文Node、3億8千万の論文間リファレンスEdgeという膨大なグラフデータセット。
  * 論文は7300万はAbstract、800万は全文が取得できる。

## Q&A

* [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
  * Stanfordの公開してる質問応答の大規模データセット
* [Maluuba News QA](http://datasets.maluuba.com/NewsQA)
  * CNNのニュース記事をベースにした質問応答のデータセット。質問数は10万overというサイズ。SQuAD同様、質問はクラウドソーシングで作成しており、回答は本文中の抜粋(一単語とは限らない)になる。しかも、答えられない可能性もあるという歯ごたえのある問題設定になっている。
* [MS MARCO](http://www.msmarco.org/)
  * Microsoftが公開した質問応答のデータセット(10万件)。質問/回答が、人間のものである点が特徴(Bing=検索エンジンへの入力なのでどこまで質問っぽいかは要確認)。回答はBingの検索結果から抜粋して作成
  * [2018/3/2に、質問数を10倍(100万)にしたV2のデータセットがリリースされた](https://twitter.com/MSMarcoAI/status/969266855633440768)。回答の質の向上にも気が払われており、情報源となるテキストを単純に抽出したような回答は適切な書き替えが行われているとのこと。
  * [MS MARCO: A Human Generated MAchine Reading COmprehension Dataset](https://arxiv.org/pdf/1611.09268v1.pdf)
* [TriviaQA: A Large Scale Dataset for Reading Comprehension and Question Answering](http://nlp.cs.washington.edu/triviaqa/)
  * 大規模なQAのデータセット(65万件)。QAだけでなく、Evidence(Answerの根拠となる複数のWebページ、またWikipedia)が付属。
  * 公開時点(2017/5)では、人間の精度80%に対してSQuADで良い成績を収めているモデルでも40%なので、歯ごたえのあるデータセットに仕上がっている。
  * [TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension](http://nlp.cs.washington.edu/triviaqa/docs/triviaQA.pdf)
* [WebQuestions/Free917](https://nlp.stanford.edu/software/sempre/)
  * 5W(When/Where/Who/What/Why)で始まる質問に対する回答を集めたデータセット。
  * WebQuestionsは学習/テスト=3,778/2,032の質問が、Free917は641/276のデータが登録されている
* [WikiTableQuestions](https://github.com/ppasupat/WikiTableQuestions)
  * テーブルを見て質問に回答するというタスクのデータセット。売上のテーブルがあったとしたら、20XX年のY事業部の売上は?などといった質問に回答する。
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
* [The NarrativeQA Reading Comprehension Challenge Dataset](https://github.com/deepmind/narrativeqa)
  * 既存のMachine Readingのデータセットは、答えがばっちり入っている一文がありその周辺の情報しか回答に必要ないものが多かった。
  * そこで、Q&Aを文から抽出する形でなく、サマリを読んで作成するという形にして、より読んだ内容をメタに理解していないと解けないデータセットを作成した。
  * ストーリー数は1,567、Q&Aの個数は46,765となっている。回答はシンプルなものとされ、単語数の平均は4.73。
* [MilkQA](http://www.nilc.icmc.usp.br/nilc/index.php/milkqa/)
  * ブラジル農牧研究公社(Embrapa)への問い合わせを匿名化した2,657のQ&Aが収録されている(なお言語はポルトガル語)。
  * 既存のQ&Aのデータセットは少ない語や選択式でこたえられるものが多く、こちらはより現実的な質問回答のデータセットになっている
  * [MilkQA: a Dataset of Consumer Questions for the Task of Answer Selection](https://arxiv.org/abs/1801.03460)
* [AI2 Reasoning Challenge Dataset](http://data.allenai.org/arc/)
  * 小学生レベルの科学に関する選択式質問のデータセット。総質問数は7,787。
  * ChallengingとEasyの2種類があり、前者は既存の情報抽出の手法・単語共起で回答できなかったものが収録されている。
* [MultiRC](http://cogcomp.org/multirc/)
  * 短いパラグラフと、それについての質問からなるデータセット。パラグラフの数は800、質問数は6000程。
  * 回答は多肢選択式で、パラグラフ中に言及がないものもある。また、パラグラフは、単一でなく7つのドメイン(ニュースや小説、歴史の文書など)から構成される。
* [QuAC](http://quac.ai./)
  * SQuADに対話的な要素を含んだ形のデータセット。対話数は14,000で、QAペア数は10万(1対話当たり7QAがある計算)。
  * 「アメリカの大統領は？」「その人の妻は？」という形で続いていくイメージ。先生と生徒という役割分担があり、生徒側はフリーフォームだが先生側はドキュメントの特定箇所を抜粋する形で回答する。
  * 先生側は対話行為タイプを生徒に提示するようにしており(さらに質問を、など)、対話が組み立てやすくなるよう工夫をしている。
* [CoQA](https://stanfordnlp.github.io/coqa/)
  * 対話形式の質問回答のデータセット。対話数は8000で、127,000のQAが含まれる。
  * ある文書について、2人のアノテーターが質問/回答を行うという形で作成されている。
  * 文書は、7つのドメイン(Wikipedia/News/Jeopardyなど)から取得されておりバリエーションに富んでいるとのこと。
* [HotpotQA](https://hotpotqa.github.io/explorer.html)
  * 複数文書にまたがった情報抽出(multi-hop)が求められるQAのデータセット。
  * multi-hopなデータセットは既存のものがあるが、知識グラフではなくWikipediaから作成されている点、また大きな差異として回答根拠になる文についてアノテーションが行われている。
  * 収集に際して単文書から回答可能な質問を一定量作成したワーカーの質問を除外、既存QAモデルで高いconfidenceで回答できる文を除外などの工夫がとられている
* [ReviewQA](http://www.europe.naverlabs.com/Blog/ReviewQA-A-novel-relational-aspect-based-opinion-dataset-for-machine-reading)
  * レビュー評価の観点に関するQAデータセット。
  * レビューがあった場合、その評価は1~5の何れか、レビュー中でXXとYYはどちらが良いとされているか、等といった質問が収録されている。
  * レビュー数は100,000、質問数は500,000。
* [Commonsense Explanations (CoS-E) Dataset](https://github.com/salesforce/cos-e)
  * 常識(Commonsense)を学習させるためのデータセット。多肢選択問題の形式をとっており、質問数は1万ほど。
  * 「なぜ人はゴシップ誌を読むのか?」=>「学習のため」「娯楽のため」「情報を得るため」・・・などといった質問が収録されている。
* [MLQA: Evaluating Cross-lingual Extractive Question Answering](https://github.com/facebookresearch/MLQA)
  * 質問回答のモデルを多言語で評価するデータセット。
  * EnglishのQAが12,000あり、他言語(6言語)のQAが各5,000ある(Englishから多言語への転移という現実的な状況を想定している)。平均4言語で同じ質問があり、これにより質問の難しさと言語の難しさを分けて評価できる。
* [TyDi QA: A Multilingual Question Answering Benchmark](https://google-research-datasets.github.io/tydiqa/tydiqa.html)
  * Googleが多言語の質問回答データセットを公開。表記や構造が異なる様々な言語を収録している。
  * データを作る際は翻訳でなくオリジナルの言語で文章を提示する、質問する内容は文章に関係なくても想起されるものであればOKなど、自然なQAに近づけるための工夫が取られている。
* [JAQKET](https://www.nlp.ecei.tohoku.ac.jp/projects/jaqket/)
  * 日本語でのクイズデータセット。質問・回答以外に回答候補が提供されている。回答はWikipediaの記事名になるように設定されている。
  * 問題文の数は13061、dev1が995、dev2が997。
* [MKQA: Multilingual Knowledge Questions & Answers](https://github.com/apple/ml-mkqa/)
  * 多言語の質問回答データセット(日本語も入っている)。解答はYes/No/テキストスパンだけでなく「回答無し」も含んだいずれかから行うようになっている。
  * 解答には可能な場合WikipediaのQIDが付与されており、知識グラフへの活用や(存在する場合)他言語への翻訳ができるようになっている
* [自治体における「子育てAIチャットボット」の普及に向けたオープンデータ](https://linecorp.com/ja/csr/newslist/ja/2020/260)
  * LINEが参画している子育てオープンデータ協議会が公開している、渋谷区のFAQを基にどの自治体でも活用できるようにしたFAQデータセット。
  * ライセンスは、クリエイティブ・コモンズ・ライセンスの表示4.0国際（https://creativecommons.org/licenses/by/4.0/legalcode.ja に規定される著作権利用許諾条件を指す。）による。
* [BeerQA](https://beerqa.github.io/)
  * 回答するためのステップ数が決まっていないQAデータセットBeerQAが公開。1stepとしてSQuAD、2stepとしてHotpotQA、3step以上の文書検索が必要な質問として新たにQAを作成し3つ複合したデータセットとなっている。ベースラインとしてTransformer型のモデルも提案。
* [JaQuAD](https://github.com/SkelterLabsInc/JaQuAD)
  * SQuADを模した、日本語のQ&Aデータセット。

## IR

* [CodeSearchNet](https://github.blog/2019-09-26-introducing-the-codesearchnet-challenge/)
  * コード検索を改善するためのデータセット。
  * コード(メソッド)とコメント、メタデータを合わせたコーパスと、検索結果についてプログラマーや研究者が関連度をアノテーションしたデータの2つが公開されている。
* [TripClick](https://tripdatabase.github.io/tripclick/)
  * 臨床研究を検索できる[Trip](https://www.tripdatabase.com/)で収集されたデータセット。検索クエリ/結果のログだけでなく、MEDLINEに登録されている文献についてタイトル/概要をセットにしたIR用のデーセット、それをDNNのモデル学習用に加工したデータセットが提供されている。
* [Amazon Multilingual Counterfactual Dataset (AMCD)](https://github.com/amazon-research/amazon-multilingual-counterfactual-dataset)
  * もし〜だったら、という半事実を含む商品レビューのデータセット。英語、ドイツ語、日本語が対象。割合は1~2%程度だが、事実に基づかないレビューのためユーザー体験が悪くなる。半事実を含む文の構文からデータを収集し、半事実ではないが似ている文をBERTの類似度で収集している。

## Reasoning

* [HolStep](http://cl-informatik.uibk.ac.at/cek/holstep/)
  * Googleから公開された、論理推論を学習するための大規模データセット。与えられた情報の中で推論に重要な点は何か、各推論間の依存関係、そこから導かれる結論は何か、などといったものがタスクとして挙げられている。
  * [HolStep: A Machine Learning Dataset for Higher-order Logic Theorem Proving](https://arxiv.org/abs/1703.00426)
* [SCONE: Sequential CONtext-dependent Execution dataset](https://nlp.stanford.edu/projects/scone/)
  * Stanfordから公開されている論理推論のためのデータセット。
  * 各シナリオには状況が設定されており(ビーカーがn個ある、絵がn個並んでいる、など)、それに対して5つの連続した操作が自然言語で記述されており(猫の絵を右にずらす、犬の絵をはずす、など)、それらを実行した場合の最後の状態を推定させるのがタスクになる。
  * [Simpler Context-Dependent Logical Forms via Model Projections](https://arxiv.org/abs/1606.05378)
* [ROCStories](http://cs.rochester.edu/nlp/rocstories/)
  * 4つの文からなるストーリをコンテキストとして、その結末を回答するというタスクのデータセット。
  * 例: 「カレンはアンとルームメイトになった」「アンはコンサートに行かない？と誘った」「カレンはそれに同意した」「コンサートは最高だった！」=>結末は「カレンとアンは親友になった」or「カレンはアンのことを嫌いになった」の何れか？(2者択一)
* [SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference](http://rowanzellers.com/swag/)
  * 自然言語による推論を行うためのデータセット。ある状況に続くシーンとして想定されるものを選択肢から選ぶ形(「車に乗り込んだ」=>「アクセルを踏んだ」というような)。
  * 工夫としては、正解以外の選択肢はモデルをだますよう人間が作成しているという点(見たところ文として自然だが意味が通らない印象)

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
* [THE PERSONA-CHAT DATASET](https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/personachat)
  * ペルソナを維持した、一貫性のある対話を行うためのデータセット。
  * クラウドソーシングを使い1155人文のペルソナ(猫を飼っている、など自分に関する5つ以上の文)を収集し、さらに語彙などが偏らないよう別の人による書き換えを行いペルソナを用意。
  * そこから無作為に選んだペルソナで対話をしてもらい、総計10,981の対話を収録している。
  * [Personalizing Dialogue Agents: I have a dog, do you have pets too?](https://arxiv.org/abs/1801.07243)
* [CoCoA - CraigslistBargain](https://stanfordnlp.github.io/cocoa/)
  * 価格交渉対話のデータセット。1402のアイテムについて、価格交渉を行っている6682対話が収録されている。
* [SearchQA](https://github.com/nyu-dl/SearchQA)
  * Jeopardy!のクイズをベースに、クイズに関連する情報を検索エンジンから集めたデータセット。決まったQAのコンテキストを、検索エンジンで補完した形になっている。
  * QA数は14万、各QAは平均49.6のコンテキストを持つ。検索クエリも含まれているため、コンテキストの情報を取り直すことも可能。
* [Natural Questions](https://ai.google.com/research/NaturalQuestions)
  * Googleの検索データに基づいたデータセットで、質問はルール(who/when/whereから始まるなど)に合致する検索クエリから取得され、回答は長短の2つが用意されている(長: Wikipediaの該当パラグラフ、単: パラグラフ内の回答そのもの)。
  * 学習データ数は307,373で各質問には1つの回答だが、validation(7,830)/test(7,842)では質問に対し複数(5つ)の回答が用意されている。
* [PolyAI-LDN/conversational-datasets](https://github.com/PolyAI-LDN/conversational-datasets)
  * 公開済みであるAmazon/Redditなどの対話データから、再現性のあるデータセットを作成するツール。百万単位のデータセットを得ることが可能で、フォーマット/train・testのsplitが適切に行われるためベンチマークとして使用できる。

## Summarization

* [DUC 2004](http://www.cis.upenn.edu/~nlp/corpora/sumrepo.html)
  * 文章要約のためのデータセット。ベースラインとなるアルゴリズムによる要約結果も収録されており、それらのROUGEスコアと比較が可能。
* [boxscore-data](http://lstm.seas.harvard.edu/docgen/)
  * バスケットボールの試合のスコアと、試合結果についての要約をペアとしたデータセット。数値表現をテキストにする試み。
  * Rotowire/SBNationといったスポーツサイトからデータを収集しており、総計約15000のペアが収録されている。
* [CORNELL NEWSROOM](https://summari.es/)
  * 要約のための大規模なデータセット。
  * 38の代表的な出版/報道局から集められた130万記事について、記事と要約がセットになっている(発行年は1998年から2017年まで)。

## Correction

* [AESW](http://textmining.lt/aesw/index.html)
  * 文書校正前後の論文を集めたデータセット。
  * 学習データには約100万文が含まれ、そのうち46万件ほどに校正による修正が入っている。
* [Lang-8 dataset](http://cl.naist.jp/nldata/lang-8/)
  * 語学学習を行うSNSであるLang-8から収集されたデータセット。Lang-8では学習している言語で作文を行うと、その言語を母国語としている人から添削を受けることができる。この学習者の作文と訂正された作文のペアがデータセットとして収録されている。
  * 10言語のデータが含まれており、総数は約58万文書に及ぶ。
  * 実はNAISTが公開しており、詳細はこちらから参照できる。[語学学習 SNS の添削ログからの母語訳付き学習者コーパスの構築に向けて](https://www.ninjal.ac.jp/event/specialists/project-meeting/files/JCLWorkshop_no6_papers/JCLWorkshop_No6_27.pdf)
* [Paralex Paraphrase-Driven Learning for Open Question Answering](http://knowitall.cs.washington.edu/paralex/)
  * WikiAnswerから収集した、質問の言いかえデータセット。WikiAnswerでは、同じ質問をマージすることができるようで、そのデータを使用している。
  * データ数は1800万

# Audio

## Sound

* [DCASE](http://www.cs.tut.fi/sgn/arg/dcase2016/task-acoustic-scene-classification)
  * 自然音の分類を行うタスク(公園の音、オフィスの音など)で、学習・評価用データが公開されている。
* [Freesound 4 seconds](https://archive.org/details/freesound4s)
  * FreeSoundの音声データとそのメタデータをまとめたデータセットが公開(普通は頑張ってAPIを叩かないと行けなかった)。音響特徴を捉えるモデルの学習に役立ちそう(以前楽器の分類の学習に使ったことがある)。
* [FSD is a large-scale, general-purpose audio dataset](https://datasets.freesound.org/fsd/)
  * FreeSoundオフィシャルのデータセット。26万件のサウンドに、階層化された600のクラスラベルが付与されている(アノテーション件数自体は60万件に上る)。
  * 音声には、楽器などの音以外に人間や動物の鳴き声など、多様な音声が含まれる。
  * このデータセットを利用した、[Kaggleのコンペティション](https://www.kaggle.com/c/freesound-audio-tagging)も開催されている。
* [AudioSet](https://research.google.com/audioset/)
  * YouTubeから抽出した10秒程度の音に、人の声や車の音といった632のラベル(人の声→シャウト、ささやき、など階層上に定義されている)が付与されている(人手で)。その数その数200万！
* [MAESTRO Dataset](https://storage.googleapis.com/magentadata/papers/maestro/index.html)
  * ピアノ演奏と対応するMIDIデータのデータセット。
  * ピアノの演奏は、インターネット上のピアノ演奏コンペティションであるInternational Piano-e-Competitionから取得されている。演奏数1,184，曲数430。
* [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove)
  * ドラム音源のデータセット。
  * 8割プロで構成された総勢10名のドラマーによる音源が収録されており、総収録時間は13.6時間、MIDIファイル数は1150にもなる。
* [ToyADMOS: A Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection](https://github.com/YumaKoizumi/ToyADMOS-dataset)
  * 異常音を検知するためのデータセットToyADMOSが公開(Toyの名の通り、おもちゃを音の収録に使っている)。
  * 固定機器としてベルトコンベア、移動機器として電車、また検査機器を使った場合として車を検査機器上で走らせた音の3種類が収録されている。
* [MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection](https://zenodo.org/record/3384388#.XYmS_ij7SUk)
  * 工場機械の異常検知を、音から行うためのデータセットMIMIIが公開。バルブやポンプ、ファンといった工場内の機械の音が収録されている。
  * 異常も、漏れやこすれなど様々なタイプをそろえている。

## Speech

* [声優統計コーパス](http://voice-statistics.github.io/)
  * 独自に構築された音素バランス文を、プロの女性声優3名が読み上げたものを録音したコーパス。
  * 3パターンの感情(通常・喜び・怒り)での読み上げが含まれる。48kHz/16bitのWAVファイルで、総長約2時間、総ファイルサイズ720MB。
* [JSUT(Japanese speech corpus of Saruwatari Lab, University of Tokyo)](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)
  * 日本語テキストと読み上げ音声からなるコーパス。一人の日本語女性話者の発音を無響室で録音。録音時間は10時間で、サンプリングレートは48kHz。
  * 常用漢字の音読み/訓読みを全てカバーするといった網羅性だけでなく、旅行ドメインのフレーズといったドメイン特化のものも収録されている。
  * 27曲の童謡の歌声データを収録したJUST-song、声真似(音声模倣)のデータを収録したJUST-viも併せて公開されている。
* [Speech Commands Dataset](https://www.tensorflow.org/versions/master/tutorials/audio_recognition)
  * TensorFlowとAIYのチームから公開された、30種類のYes, No, Up, Downなどといった短い音声による指示/応答を集めたデータセット。総数は65,000。
  * このデータセットを利用した音声認識モデルの構築手順が、TensorFlowのチュートリアルとして提供されている。
* [The Spoken Wikipedia Corpora](http://nats.gitlab.io/swc/)
  * Wikipediaの記事を読み上げたデータセット。音声と単語の対応、単語と文中語の対応がアノテーションされている(単語がfive, hundredだった場合、文中語の500に対応、など)。
  * しかも多言語のデータセットで、英語・ドイツ語・オランダ語が提供されている。
* [Common Voice](https://voice.mozilla.org/data)
  * Mozillaが公開した、音声認識のためのデータセット。音声データは500時間分、2万人以上から録音という世界で二番目の規模。
  * モデルも公開されている: [DeepSpeech](https://github.com/mozilla/DeepSpeech)
* [VoxCeleb2: Deep Speaker Recognition](http://www.robots.ox.ac.uk/~vgg/data/voxceleb2/)
  * 6112名の著名人の、100万発話を収集したデータセット。収集は、顔認識のモデルを使いほぼ自動で行われている。
  * 具体的には、VGGFace2に登録されている著名人のインタビュー動画をYoutubeからダウンロードし(人名+interviewで検索しダウンロード)、動画中の顔を認識＋話者推定(音声と唇の動きから推定するSyncNetを使用)を行い該当箇所の音声を切り出すという手法。
* [AVSpeech: Audio Visual Speech dataset](https://looking-to-listen.github.io/avspeech/explore.html)
  * YouTubeの講義動画などから収集した、15万人以上、4700時間分の明瞭な発話及び話者動画データセット。
  * 1つの動画は3~10秒からなり、単一の話者の顔が映りこんでいてかつ背景の雑音がない明瞭な音声のものが選ばれている。
* [JVS-MuSiC](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music)
  * 100名の歌唱者による歌唱コーパス。各歌唱者の発話データは[JVS corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)に収録されている。
* [CoVoST V2](https://ai.facebook.com/blog/covost-v2-expanding-the-largest-most-diverse-multilingual-speech-to-text-translation-data-set/)
  * 多言語の音声認識データセット。Mozilla’sの公開するCommon Voiceデータセットをもとに、21言語=>英語、英語=>15言語のコーパスを作成している(収録時間は2,900時間)。
* [JSSS: Japanese speech corpus for summarization and simplification](https://sites.google.com/site/shinnosuketakamichi/research-topics/jsss_corpus)
  * 比較的長い文/平易な文などの音声が収録された日本語音声コーパス。
  * 時間制約付きの音声要約・平易な日本語・短い文・長文の4種類が収録されている。それぞれライセンスが異なるため扱う際は要チェック。

## Music

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
* [MUSDB18 dataset](https://sigsep.github.io/musdb)
  * 音源分離を行うためのデータセット。マルチトラックのmp4でエンコードされており、0がAll track、1がトラム、2がバス、3がその他楽器、4がボーカルに割り振られている。
  * 学習用に100曲、テスト用に50曲が提供されている。
* [The NES Music Database](https://github.com/chrisdonahue/nesmdb)
  * ファミコンのゲーム楽曲を収録したデータセット。397タイトルの計5278曲が含まれる。
  * アセンブリから楽曲再生にかかわる部分を抽出し(いいのか？)、MIDIを始めとした扱いやすい形式に変換している。制限された音階/楽器をうまく組み合わせているので、学習に良いとのこと。

# Other

* [grocery-shopping-2017](https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2)
  * Instacartという食品のオンラインショップの購買データのデータセットが公開。時間のデータも、時間(0~24時)だけながら利用可能。
* [気象庁：過去の気象データ](http://www.data.jma.go.jp/gmd/risk/obsdl/index.php)
  * 地点毎になるが(複数選択可能)、過去の気象データをCSV形式でダウンロードできる。
* [Global Terrorism Database](https://www.kaggle.com/START-UMD/gtd)
  * 1970~2016年(なぜか2013年だけない)の間の世界で発生したテロ事件のデータセット。その数17万件。
  * STARTというテロ対策を研究する団体がメンテナンスを行っており、特徴量として発生した場所や犯人や手口などが含まれている。
* [THE STANFORD OPEN POLICING PROJECT](https://openpolicing.stanford.edu/)
  * スタンフォードが取り組んでいる、法的機関(警察)による交通取り締まりのオープンデータプロジェクト。
  * データには、違反日時や場所、ドライバーの年齢や性別などが記録されている(Jupyterのサンプルも公開されている)。
* [Microsoft/USBuildingFootprints](https://github.com/Microsoft/USBuildingFootprints)
  * Microsoftが公開した、アメリカの50の州における建物のfootprint(建物の占有領域をセグメンテーションしたようなもの)データ。OpenStreetMapをベースに作成されている。
  * Microsoftはこのデータを利用して、[セグメンテーション=>ポリゴン化を行う研究を行っている](https://blogs.bing.com/maps/2018-06/microsoft-releases-125-million-building-footprints-in-the-us-as-open-data)。
* [DataShop](https://pslcdatashop.web.cmu.edu/)
  * Pittsburgh Science of Learning Centerが公開している、教育用ソフトと生徒とのインタラクションデータ
  * 350を超えるデータセットが公開されている。
* [eQuake-RC](http://equake-rc.info/)
  * これまで発生した地震のデータセット。アップロードによる更新も可能(ただMatlab形式)。
  * 東日本大震災・熊本地震も含まれており、日本の研究者によりきちんと更新されている。
  * [Googleがこちらのデータを使用し地震発生後の余震を予測しようと試みている。](https://www.blog.google/technology/ai/forecasting-earthquake-aftershock-locations-ai-assisted-science/)
* [センター試験XMLデータ](https://21robot.org/dataset.html)
  * 東ロボプロジェクトで使用されたデータセット。各教科のテスト問題が、XML形式で収録されている。
  * ただ、著作権の問題からか国語の問題、またいくつかの科目での画像が含まれない。
* [CRCNS - Collaborative Research in Computational Neuroscience](https://crcns.org/data-sets)
  * 脳の神経活動を記録したデータセットの公開サイト。
* [RoboNet: Large-Scale Multi-Robot Learning](https://github.com/SudeepDasari/RoboNet)
  * 7種類のロボットで記録した1500万のビデオフレームのデータセット。様々なロボットの軌跡データから表現学習(次フレーム予測/過去フレーム予測)を行うことで、Zero/Few shotが可能な転移性能の高いモデルを構築できる。
* [LEAF](https://leaf.cmu.edu/)
  * 分散学習(Federated Learning)のためのベンチマークデータセット。MNISTやTwitterのSentiment Analysisの学習を複数分散デバイス(ユーザー)想定で行ってみることができる。
* [Geolonia 住所データ](https://github.com/geolonia/japanese-addresses)
  * 日本の正式住所名のデータセット。緯度経度情報が付属しているので、名寄せを行うことで地図上に配置することができる。
* [WILDS](https://wilds.stanford.edu/)
  * 現実世界で起こる特徴シフトへの対応を評価するためのベンチマーク。画像では建物がなくなる、農地が団地になる、動物が現れるといった変化、自然言語ではレビューやコメントの変化などが扱われている。(学習外の)分子構造の予測などもある。

## Chemical

* [MoleculeNet](https://arxiv.org/abs/1703.00564)
  * MoleculeNetという、新薬発見のための分子・分子物理・生体物理・生体？という4種類のデータを包含したデータセットが公開。
  * [DeepChem](https://github.com/deepchem/deepchem)という化学特化のライブラリに組込済
* [Tox21](https://tripod.nih.gov/tox21/challenge/data.jsp)
  * 化学化合物の構造からその毒性(toxic effects)を推定するタスクのためのデータセット。化合物数は12,000、毒性は12の毒性についての値が用意されている。
* [QM dataset](http://quantum-machine.org/datasets/)
  * 有機分子を収録したGDBというデータセットのサブセットとしてリリースされているデータセット(※GDBに収録されている分子はどんどん増加しており
  、GDB-13では10億、GDB-17では1660億ととんでもない数になっている)。
  * QM7: 原子数23の分子に制限したデータセット(総分子数は7165)。13の特性をデータに足したQM7bもある。
  * QM8: CONF(炭素・酸素・窒素・フッ素)原子8つまでで構成される、合成しやすい有機分子20,000の特性が収録されたデータセット。
  * QM9: CHONF(炭素・水素・酸素・窒素・フッ素)で構成される13万の有機分子の特性が収録されたデータセット。
* [Alchemy Dataset](https://alchemy.tencent.com/)
  * 有機化合物のデータセットが公開。収録分子数はQM9とほぼ同程度だが、構成原子の種類として硫黄(S)/塩素(CI)などが追加されている。
* [dSPP: Database of structural propensities of proteins](https://peptone.io/dspp)
  * タンパク質(アミノ酸の鎖のベクトル)から構造的傾向スコア(structural propensity score)を予測するためのデータセット。
  * Kerasから使うためのユーティリティも提供されている([dspp-keras](https://github.com/PeptoneInc/dspp-keras))。

## Security

* [SARD Dataset](https://samate.nist.gov/SRD/testsuite.php)
  * SARD(Software Assurance Reference Dataset)にて提供されている、ソフトウェアの脆弱性を検証するためのデータセット
  * 脆弱性を含むC/Javaのコードなどが提供されている
* [PHP Security vulnerability dataset](https://seam.cs.umd.edu/webvuldata/data.html)
  * PHPのアプリケーションの脆弱性(CVEのIDなど)と、そのコードから抽出された機械学習で利用する特徴量のデータセット。PHPアプリケーションはPHPMyAdmin、Moodle、Drupalの3点
* [Passwords](https://wiki.skullsecurity.org/Passwords)
  * パスワードの辞書、またこれまでにサイトから流出したパスワードのデータセットがダウンロードできる
* [EMBER: Endgame Malware BEnchmark for Research](https://github.com/endgameinc/ember)
  * 悪意あるWindowsのPortable Executable file(PE file)を検知するためのデータセット。件数は100万近くあり、特徴抽出済み。
  * 特徴抽出のスクリプトは公開されており、このため自分で集めたデータで拡張することが可能。

## Reinforcement Learning

* [GoGoD](http://senseis.xmp.net/?GoGoDCD)
  * プロの囲碁棋士の対局データセット。85,000局分が含まれており、お値段は15USD
* [wangjinzhuo/pgd](https://github.com/wangjinzhuo/pgd)
  * プロの囲碁棋士の対局データセット。GitHub上でフリーで公開されており、約25万局が収録されている。
* [TorchCraft/StarData](https://github.com/TorchCraft/StarData)
  * StarCraftのプレイデータ。約6万5千プレイ、フレーム数にして15億(!!)という大規模なデータセット。
* [RoboTurk](http://roboturk.stanford.edu/realrobotdataset.html)
  * ロボットアームによるマニピュレーションを記録したデータセット。教師あり学習や模倣学習などに使用することができる。
  * 既存のデータセットは自己教師のものが多かったが、こちらは実際人が操作した画像、54名による2114の物体移動/配置デモが記録されている。収録時間はのべ111時間にも及ぶ。
* [RL Unplugged: Benchmarks for Offline Reinforcement Learning](https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged)
  * (実環境にアクセスしない)オフライン強化学習のベンチマークデータセット。複雑な連続値コントロールからAtariまで幅広い行動空間、探索難易度の環境のデータが収録されている。

## Climate Change

* [Climate Change Laws of the World](https://climate-laws.org/)
  * 気候変動に関する法律や訴訟の文書データセット。法律での規制に意味があるのか?の検証に使用できる。データの可視化では、法律が多いほど排出量は抑えられているように見える。
* [PyRain](https://github.com/FrontierDevelopmentLab/PyRain)
  * 降水量に関わるデータセットを素早くロードできるライブラリ。ヨーロッパ中期予報センター(ECMWF)が公開するSimSatとERA5、NASAが公開するIMERG、の3つが使用可能で、高速にデータロード可能+異なるデータを組み合わせたモデルが作成できる。
* [OGNet](https://stanfordmlgroup.github.io/projects/ognet/)
  * 衛星画像(NAIP: National Agriculture Imagery Program)を500 x 500ピクセルに切り分け、石油精製施設かどうかを判定したデータセット。画像数7,066のうち、石油精製施設が149が含まれる。

# Dataset Summary Page

* [kaggle](https://www.kaggle.com/)
  * データ解析のコンペティションサイト。モデルの精度を競い合うことができ、データも提供されている。[Kaggle Datasets](https://www.kaggle.com/datasets)でデータの検索、また公開もできるようになった。
* [NLP-progress](https://github.com/sebastianruder/NLP-progress)
  * 自然言語処理の各タスクにおける、精度のランキングをまとめたサイト(リポジトリ)。
  * 各タスクでベンチマークとして使用されているデータセットもまとまっている。
* [人文学オープンデータ共同利用センター](http://codh.rois.ac.jp/)
  * 日本の古典(徒然草や源氏物語)の書籍画像、また本文テキストなどのデータを公開しているサイト。中にはレシピ本などの面白いものもある。
  * 機械学習への応用をきちんと想定しており、古文字の画像認識用データセットなども公開している。
* [国立情報学研究所](http://www.nii.ac.jp/dsc/idr/datalist.html)
  * 日本国内で公開されているデータセットはたいていここを見れば手に入る。ただ研究用途のみで申請書が必要。
* [Harvard Dataverse](https://dataverse.harvard.edu/)
  * ハーバード大学が公開している研究データのセット。自然音のクラス分類のデータ(ESC)などがある。
* [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.html)
  * 機械学習のためのデータセットを集めているサイト。
* [Microsoft Research Open Data](https://msropendata.com/)
  * Microsoftが公開したオープンデータを検索、ダウンロードできるサイト。Azure連携機能があり、選択したデータセットをすぐに配置可能。
* [20 Weird & Wonderful Datasets for Machine Learning](https://medium.com/@olivercameron/20-weird-wonderful-datasets-for-machine-learning-c70fc89b73d5)
  * 機械学習で使えるデータセットのまとめ集。UFOレポートとか面白いデータもある。
* [自然言語/音声認識学習用データのまとめ](http://qiita.com/icoxfog417/items/44aeb9486ce1b7130f76)
* [Microsoft Azure Marketplace](https://datamarket.azure.com/browse/data)
  * NFLの試合結果や人口統計など、様々なデータが提供されている(有料なものもあるたが、無料も多い)。
* [ikegami-yukino/dataset-list](https://github.com/ikegami-yukino/dataset-list/blob/master/free_text_corpus.md)
  * 日本語・英語のテキストコーパスのまとめ
* [beamandrew/medical-data](https://github.com/beamandrew/medical-data)
  * 機械学習のための化学系のデータセットのまとめ
* [Web Technology and Information Systems](https://www.uni-weimar.de/en/media/chairs/computer-science-and-media/webis/corpora/)
  * Web Technology and Information Systemsの研究で使用されたコーパス集
* [niderhoff/nlp-datasets](https://github.com/niderhoff/nlp-datasets)
  * 自然言語系のデータセットがまとめられたページ。更新も割と行われている。
* [The Extreme Classification Repository: Multi-label Datasets & Code](http://manikvarma.org/downloads/XC/XMLRepository.html)
  * 多量のラベルを予測するExtreme Classificationにおけるデータセットをまとめたページ(EURLexやAmazonCat、Wiki10など)。データセット以外に、代表的手法の性能評価や実装へのリンクがある。
* [Google Dataset Search](https://toolbox.google.com/datasetsearch)
  * Googleが公開するデータセットを検索するためのサービス。
* [VisualData](https://www.visualdata.io/)
  * 画像系のデータセットがまとめられたサイト。カテゴリやキーワードで検索することができる。
* [The Big Bad NLP Database](https://datasets.quantumstat.com/)
  * 自然言語処理のデータセットをまとめたサイト。SQL-to-Textなど細かいタスクのデータセットも掲載されている。

# To make your own

* [ヒューマンコンピュテーションとクラウドソーシング ](https://www.amazon.co.jp/dp/4061529137)
* [Crowdsourcing (for NLP)](http://veredshwartz.blogspot.jp/2016/08/crowdsourcing-for-nlp.html)
  * データを集めるのに欠かせない、クラウドソーシングの活用方法についての記事。クラウドソーシングに向いているタスク、信頼性担保の方法、料金についてなど実践的な内容が紹介されている。
* [Natural Language Annotation for Machine Learning](http://shop.oreilly.com/product/0636920020578.do)
* [バッドデータハンドブック ―データにまつわる問題への19の処方箋](https://www.amazon.co.jp/dp/4873116406)
* [ガラポン](http://garapon.tv/developer/)
  * APIでテレビの字幕のデータを取ることができるらしい
