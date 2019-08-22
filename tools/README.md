# Tools

機械学習モデルの実装を行う際に役立つツールについてまとめています。  
なお、深層学習系のフレームワーク(TensorFlow/Chainerなど・・・)は既知だと思うので、その当たりは除外しています。

# Vision

* [OpenFace](https://cmusatyalab.github.io/openface/)
  * 画像から顔を認識し、128次元の特徴量を算出するツール。この特徴を利用し、独自の分類機などを作ることができる。
  * 顔の検知には、OpenCV/dlibを使用
  * Apache 2.0 License
* [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
  * 体のキーポイント(関節など)を検知するためのライブラリ
  * OpenCV/Caffeベース
  * ライセンスは独自のもので、商用利用については応相談
* [self-driving-car](https://github.com/udacity/self-driving-car/)
  * Udacityで自動運転車の講座が開講されるに伴い公開された、オープンソースの自動運転車の実装
  * 他の公開されている実装としては、[comma.ai](https://github.com/commaai/research)、Baiduが公開した[Appolo](http://apollo.auto/index.html)などがある。
* [europilot](https://github.com/marshq/europilot)
  * オープンソースの自動運転シミュレーター。Euro Truck Simulatorというトラックシミュレーションゲームと接続して動作する。
* [YOLO](https://pjreddie.com/darknet/yolo/)
  * You only look once (YOLO)と名付けられた、リアルタイムでの物体認識を行うソフトウェア。
  * C言語性のDNNのフレームワーク[darknet](https://pjreddie.com/darknet/)で構築されている
* [OpenZoonz](https://opentoonz.github.io/)
  * 2Dアニメーションの制作ソフトウェア(ジブリでの使用実績もあり)だが、ラスタライズ画像をベクター画像に変換できるという機能がある。
  * sketch-RNNなど、ベクターデータが必要な場合に役に立つ。
* [Detectron](https://github.com/facebookresearch/Detectron)
  * Facebookの公開した物体検出を行うためのオープンソースライブラリ。Caffe2ベースで稼働する
  * Apache 2.0ライセンスで使用可能。
* [rpg_esim](https://github.com/uzh-rpg/rpg_esim)
  * 画像における光度変化をイベントとして捉えて描画するイベントカメラをシミュレートするオープンソース。
  * C/C++で実装されておりリアルタイムで使用することが可能。
* [Runway](https://runwayml.com/)
  * デザイナーのための機械学習プラットフォーム。学習済みの機械学習モデルを簡単に実行できるほか、ArduinoやUnityに組み込むことができる。

# NLP

* [NLTK](http://www.nltk.org/)
  * 形態素解析や品詞タグ認識など、自然言語処理に関する基礎的な処理を行うライブラリ。微妙に日本語にも対応している。
* [spaCy](https://spacy.io/)
  * NLTKど同様だが、固有表現抽出などの機能も搭載している
  * 最大のポイントとしてはエンタープライズでの利用を想定しており、マルチスレッドを用い高速に動作するよう設計されている
* [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)
  * Stanfordが提供する自然言語処理のライブラリ。基礎的なライブラリでできることは網羅されており、照応表現(彼=○○さん、とか)の解釈や、構文解析結果を利用した処理なども可能(名詞を対象にｘｘするなど)
  * 多様なプログラミング言語へのインタフェースを持つことも特徴で、[ClojureやErlang/Elixirなどまでサポートされている](https://stanfordnlp.github.io/CoreNLP/other-languages.html)
  * なお、CoreNLPも含めたStanfordが開発しているツールは[Stanford NLP Software](https://nlp.stanford.edu/software/)にまとまっている。自然言語系で何かないかなと思ったら、まず目を通してみると良い。
* [ParlAI](https://github.com/facebookresearch/ParlAI)
  * Facebookが発表した、対話モデルを開発するためのフレームワーク。
  * SQuADやbAbI、Ubuntu Dialogなど有名な質問応答/対話系のデータセットが簡単に扱えるようになっている
  * モデル評価のためにAmazon Mechanical Turkとの連携も兼ね備えるなど、対話研究を行うならとても便利なフレームワークになっている。
* [Rasa](https://github.com/RasaHQ)
  * 対話ボットが作成できるオープンソースのフレームワーク。対話の意図や固有表現の認識といった、発話意図を読み取るための機能が実装されている
  * Fine Tuningが可能で、自前のデータで学習させることができる。
* [NeuroNER](http://neuroner.com/)
  * TensorFlowベースの固有表現抽出ツール。学習済みモデルが利用できるほか、追加データによる学習も可能。
* [OpenNMT](http://opennmt.net/)
  * オープンソースの機械翻訳の実装。Torch/PyTorchで実装されている。
  * 幾つかの言語間では、学習済みモデルも提供されている。
* [AllenNLP](https://allennlp.org/)
  * PyTorchベースのNLPモデル構築のためのライブラリ
* [GluonNLP](https://github.com/dmlc/gluon-nlp)
  * MXNetベースのNLPモデル構築のためのライブラリ
* [nlp-architect](https://github.com/NervanaSystems/nlp-architect)
  * TensorFlow(Keras)/DyNetベースのNLPモデル構築のためのライブラリ
* [marian](https://github.com/marian-nmt/marian)
  * C++による高速な機械翻訳実装
* [fairseq](https://github.com/pytorch/fairseq)
  * Facebookが2017年に発表したCNNによる翻訳モデルの実装を、Torch=>PyTorchに実装しなおしたもの。
  * Transformerなどその他の翻訳/生成モデルも実装されている。
  * また事前学習済みモデルの提供、分散GPU/FP16での学習対応など活用しやすくする機能も実装されている。
* [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)
  * 先進的なEncoder-Decoderモデルを集めたモデル集。モデル以外にハイパーパラメーターの設定も収録されている。
  * 翻訳のモデル(Transformer)で話題になったが、画像のEncoder-Decoderを行うことも可能。翻訳については、Attentionのビジュアライザも搭載している。
* [Jack the Reader - A Machine Reading Framework](https://github.com/uclmr/jack)
  * Machine Comprehensionを実装するためのフレームワーク。入力・モデル・出力というシンプルな3モジュール構成。TensorFlowとPyTorch双方をサポートしている
  * QA、推論(NLI)、Link Predictionをサポートしており、評価用データセットをダウンロードする仕組みもある。
* [SentEval](https://github.com/facebookresearch/SentEval)
  * 文のベクトル表現の品質を計測するためのプラットフォーム。SentimentやQuestion-typeの推定など、様々なタスクで評価ができる。
* [sumEval](https://github.com/chakki-works/sumeval)
  * 文要約の評価指標であるROUGE/BLEUを測定することのできるツール
* [FuzzyWuzzy](https://github.com/seatgeek/fuzzywuzzy)
  * 文の類似度スコアを計算してくれるツール(レーベンシュタイン距離をベースにしているよう)。表記ゆれなどを解消する際に役に立つ。
  * なぜかRustの実装もある。
* [chazutsu](https://github.com/chakki-works/chazutsu)
  * 自然言語関係のデータセットを簡単にダウンロードし、使えるようにできるツール
  * データをダウンロード、展開して、整形して・・・という作業は以外と時間がかかるもので、その個所をスキップできる
* [chakin](https://github.com/chakki-works/chakin)
  * 単語の分散表現を簡単にダウンロードできるツール。こちらも探してくるのが面倒なので、これを利用すれば簡単に利用できる。
* [neulab/compare-mt](https://github.com/neulab/compare-mt)
  * 自然言語処理における、生成系のタスクのスコアを測れるライブラリ。機械翻訳がメインターゲットだが、要約や対話生成の評価も可能なよう。

**Additional**

* [日本語自動品詞分解ツール](http://tool.konisimple.net/text/hinshi_keitaiso)
  * オンラインでさっと品詞分解ができるツール。手軽なのでよく使う

# Audio

* [Magenta](https://github.com/tensorflow/magenta)
  * Googleが開発を進める、機械学習をアートに適用するためのプロジェクト。音楽生成や、スタイルトランスファーなどのコードが組み込まれている
  * 特に音声系では、MIDIファイルを扱うためのAPIなども提供されているので、単純にデモを動かすだけでなく開発のベースとしても利用ができる。
* [librosa](https://github.com/librosa/librosa)
  * 音声データを読み込み、特徴量(MFCCなど)を算出するなどを行ってくれるライブラリ。tempoの推定なども行える。
* [sprocket](https://github.com/k2kobayashi/sprocket)
  * 統計的音声変換を行うライブラリ。Python3ベース。
  * GMMを用いた基礎的な手法が実装されており、Voice Conversion Challenge2018でベースラインシステムとして採用されている。
* [wav2letter++](https://github.com/facebookresearch/wav2letter/)
  * Facebookから公開された、CNNベースの音声認識ライブラリ。
  * 独自開発されたC++ベースの機械学習ライブラリ[flashlight](https://github.com/facebookresearch/flashlight)上で動作する。
* [nnmnkwii](https://github.com/r9y9/nnmnkwii)
  * 音声合成システムの開発を支援するライブラリ(PyTorchベース)。既存の[Merlin](https://github.com/CSTR-Edinburgh/merlin)に比べて、モデルのカスタマイズが行いやすいよう設計されている。

# Reinforcement Learning

## OpenAI Gym Families

* [OpenAI Gym](https://gym.openai.com/)
  * 強化学習モデルのトレーニング環境を提供するフレームワーク
  * OpenAI Gym自体にも多くの環境(Atariのゲーム)などが入っているが、Open AI GymのAPIを備えたサードパーティー製の環境も追加でインストールできる([gym_pull](https://github.com/openai/gym/wiki/Environments#gym_pull)参照)。
* [OpenAI Universe](https://universe.openai.com/)
  * VNCを利用し、まさに人間が操作するようにPCを操作させられる強化学習のプラットフォーム。これによって、画面操作をエージェントに学習させることができる。
  * **OpenAI Gym Retroのリリースに伴いリポジトリの更新が停止された**。
* [μniverse](https://github.com/unixpickle/muniverse)
  * OpenAI Universeの重たい所(VNCでリモート操作)、難しい所(画面がFlashで読み取りむずい)を改善したmuniverseが開発中とのこと。こちらはHTML5のゲームに限定することで、上記の問題点を回避している。
  * なお、2018年時点で既に更新されていない。
* [SerpentAI](https://github.com/SerpentAI/SerpentAI)
  * 手元のゲームを強化学習のための環境に使えるというライブラリ。Universeへの不満から生まれたとのこと。
  * ネイティブで動作し、ゲームに接続するAPIを作る機能が内包されているためゲーム側の対応を待つ必要がない。チュートリアルではSteamのゲームを使う手順を公開している。
* [EssexUniversityMCTS/gvgai](https://github.com/EssexUniversityMCTS/gvgai)
  * 簡単にゲームが作れてOpenAI Gymと統合できるツール。
  * これにより、強化学習のテストを行う際目的に応じたゲームを作成することができる。
  * ただゲームはJavaで作る必要がある。同じ目的なら[Python Learning Environment](https://github.com/ntasfi/PyGame-Learning-Environment)を模してPyGameで作ってほうがましかもしれない。

### gym envs

* [Gym Retro](https://blog.openai.com/gym-retro/)
  * 対応タイトル数1000というゲーム用の学習環境。収録されていないゲームと統合したり、プレイを録画/記録できるIntegration Toolも公開されている。なお、ゲームを扱うには当然ながらROMが必要。 
* [Rocket Lander OpenAI Environment](https://github.com/arex18/rocket-lander)
  * ロケット打ち上げがシミュレートできる環境が、OpenAI Gymで使える環境として登場。SpaceXのFalcon 9で得られたデータを元にしているようで、スラスターの角度を制御しロケットの打ち上げに挑戦できる(コントロールは連続値)。ベースラインとしてDDPGの実装が提供されている。 
* [gym-duckietown](https://github.com/duckietown/gym-duckietown)
  * OpenAI Gym対応の自動運転車シミュレーション環境が登場。シミュレーターベースのSimpleSim-v0と、物理デバイス(ラズパイを組み込んだラジコンであるDuckietown)と接続して遠隔操作ができるDuckiebot-v0の2つを提供している。
* [Gym Robotics](https://gym.openai.com/envs/#robotics)
  * OpenAI Gymに標準搭載されたロボットシミュレーション用環境。物理シミュレーターのMuJoCoと連携して動作する。
* [pybullet-gym](https://github.com/benelot/pybullet-gym)
  * 物理シミュレーターである(Py)Bulletを使用した、ロボティクスなどの連続値コントロール環境。具体的には、Gym MuJoCoとDeepMind Control Suiteに収録されている環境をエミュレートしたものが扱える。なぜすでにある環境をエミュレートしているのかというと、MuJoCoが実は商用ライセンスのソフトウェアで、購入する必要があるため。
* [roboschool](https://github.com/openai/roboschool)
  * OpenAI公式のロボットシミュレーション環境。前述の事情から、MuJoCoベースではなくBulletベースに切り替えている。マルチプレイヤーのゲームが搭載されている。
* [Learning to run](https://github.com/stanfordnmbl/osim-rl)
  * 観測情報から足の筋肉を操作するモデルを構築し、歩行を学習させるための環境。OpenAI Gym + Kerasで構築されている。
* [gym-ple](https://github.com/lusob/gym-ple)
  * PyGame製のゲーム集[PyGame-Learning-Environment](https://github.com/ntasfi/PyGame-Learning-Environment)をOpenAI Gymで学習させるためのプラグイン。
* [multiagent-particle-envs](https://github.com/openai/multiagent-particle-envs)
  * OpenAIが公開した、マルチエージェントの学習に利用可能な環境。ボール同士(赤 vs 緑)の追いかけっこゲームのような形になっており、赤は共同で追ったほうが、緑は二手に分かれて逃げたほうがいい、というようなことを学習する。
  * [実装例はこちら](https://github.com/rohan-sawhney/multi-agent-rl)
* [Competitive Multi-Agent Env](https://github.com/openai/multiagent-competition)
  * 複数のエージェントを競争させられる強化学習環境。OpenAI GymとMuJoCoをベースに稼働する。こちらの論文で使用されている: [Emergent Complexity via Multi-Agent Competition](https://arxiv.org/abs/1710.03748).
  * provides multiple scenes for multiple agents to compete. code base for the paper [Emergent Complexity via Multi-Agent Competition](https://arxiv.org/abs/1710.03748).
* [HoME-Platform](https://github.com/HoME-Platform/home-platform)
  * 家の中でエージェントを学習させられる学習環境。
  * 環境内のオブジェクトにはセグメンテーションや「棚の上の花瓶」といった言語情報がアノテートされており、音響環境も整えられている。もちろん物理エンジン搭載で、オブジェクトの移動などもシミュレートできる。
  * OpenAI Gymに対応
* [House3D](https://github.com/facebookresearch/House3D)
  * SUNCGをベースにした、屋内で動作するエージェントを学習させるための環境。
  * こちらもOpenAI Gymベース。
* [SenseAct](https://www.kindred.ai/senseact)
  * OpenAI Gym的なシミュレーション環境を、物理の世界へ拡張しようという試み。具体的には、一般に販売されているロボットを使ってタスクを定義し、それをOpenAI Gymライクなインタフェースで操作できるようにしている。
* [MAMEToolkit](https://github.com/M-J-Murray/MAMEToolkit)
  * アーケードゲームを強化学習環境として使えるようにするツールキット。当然ROMは購入する必要があるが(無料のものもいくつか紹介されている)、デモではストリートファイター3を使った学習を行っている。
* [Nintendo Learning Environment](http://olab.is.s.u-tokyo.ac.jp/~kamil.rocki/nintendo/)
  * ゲームボーイのゲームを、強化学習の環境として使えるようにした話(もちろん、実際動作させるにはROMが必要)。
  * 強化学習における学習を高速化するために、エミュレーターの実行速度を上げるというアプローチをとっている。
  * OpenAI Gymライクなインターフェースで使える。
* [Obstacle Tower Environment](https://github.com/Unity-Technologies/obstacle-tower-env)
  * Unityで作成された、塔を登っていくタイプのゲーム環境。
  * 各フロアでは上に登るための階段を見つけるのが目的で、フロア内の部屋には様々なタスクが用意されているという構成。
* [BTGym](https://github.com/Kismuz/btgym)
  * 投資におけるバックテスト(過去の値動きに対して、ある戦略がどれだけ有効かを調査する)を行うためのライブラリbacktraderをOpenAI Gymライクなインタフェースで使えるようにしたライブラリ。
* [stable-baselines](https://github.com/hill-a/stable-baselines)
  * よりユーザーフレンドリーなOpenAI Baselineの実装。
  * 各アルゴリズムについて、きちんとドキュメント化されていたりカスタムの実装と比較したりできる。また、学習済みのエージェントも提供されている。

## Simulator Integration

* [mujoco-py](https://github.com/openai/mujoco-py)
  * 物理エンジンであるMujocoをPythonから操作できるライブラリ  
* [pybullet](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet)
  * Pythonから使える物理シミュレーター。OpenAI Gym/TensorFlowにも対応していて、これらを利用した歩行トレーニングを行う[チュートリアルも提供されている](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3)。
  * [pybullet_robots](https://github.com/bulletphysics/pybullet_robots): 環境内で、実際のロボットを模したエージェントを使うことができる。よく使われるTurtlebot以外に、Boston DynamicsのAtlasやBotlabなどもある。
* [Dart](http://dartsim.github.io/)
  * ロボットの動きや、動作アニメーションを作成するためのデータ構造やアルゴリズムを提供するためのライブラリ。C++制だが、Pythonバインディング([pydart2](https://github.com/sehoonha/pydart2))が提供されている。
  * This libraray provides data structures and algorithms for kinematic and dynamic applications in robotics and computer animation. Its python bindings are [here](https://github.com/sehoonha/pydart2), examples are [provided](https://github.com/sehoonha/pydart2/tree/master/examples).
* [Gazebo](http://www.gazebosim.org/)
  * 3Dロボットシミュレーターで、使用する物理エンジンを切り替えることができる(ODE, Bullet, Simbody, DART)。豊富な[チュートリアル](http://gazebosim.org/tutorials)が提供されている。
  * Gazebo is a robot simulation toolkit that supports the ODE, Bullet, Simbody and DART physics engines. Tutorials are [provided](http://www.gazebosim.org/tutorials).
* [Unity ML - Agents](https://github.com/Unity-Technologies/ml-agents)
  * Unityが公式に公開した、Unityで強化学習を行うための環境・エージェントを作成できるSDK
  * [Unity: A General Platform for Intelligent Agents](https://arxiv.org/abs/1809.02627)

## Others

* [ELF (Extensive, Lightweight and Flexible platform for game research)](https://github.com/facebookresearch/ELF)
  * Facebookが公開したゲームで強化学習を行うためのフレームワーク
  * ゲームの実行部分はC++のスレッドで高速に、学習部分はPythonでという形で役割分担している。
  * また、PyTorchベースのベースライン実装を搭載している。
  * 2019年2月からは、[AlphaGoZero/AlphaZeroのオープンソース実装として再スタートしている](https://github.com/pytorch/ELF)
* [dm_control](https://github.com/deepmind/dm_control)
  * DeepMind製の強化学習の学習環境。OpenAI GymよりもContinuous Controlなタスク(アームの上げ下げやステアリングといった、Action空間が連続的(実数値)になるタスク)を取り揃えており、モデルの評価環境として使える。ベースラインモデルも提供されている。
  * [DeepMind Control Suite](https://arxiv.org/abs/1801.00690)
* [Ray](https://github.com/ray-project/ray)
  * Berkeleyが公開した、分散実行を行うためのライブラリ。
  * 動的グラフのサポート、環境/パラメーターの共有のサポートなど、分散学習に必要と思われる機能は全部入りになっている。
  * しかも、タスクとして分散実行が必要なことが多い強化学習、ハイパーパラメーター探索には専用フレームワーク(それぞれRay RLlib/Ray.tune)が用意されている
* [Microsoft/AirSim](https://github.com/microsoft/airsim)
  * Microsoftが公開したシミュレーター。自動運転や、ドローンといった実世界で動かすようなものの学習を行うことができる。
* [malmo](https://github.com/Microsoft/malmo)
  * 協調するマルチエージェントの学習に特化した学習プラットフォーム。
  * Microsoftが公開しており、Minecraftの上で動く
  * Python、Lua、C#、C++、Javaなど多様なインタフェースを持つ
  * NeurIPS 2019のコンペティションに際して、Malmoをベースに使いやすくした[MineRL](https://github.com/minerllabs/minerl)が公開された。
* [VINE](https://github.com/uber-common/deep-neuroevolution)
  * Uberが公開した進化戦略の学習過程を可視化するためのツール
  * 各世代のスコア、また挙動の変化を確認できる。
* [Pommerman](https://www.pommerman.com/)
  * ボンバーマンを模した、マルチエージェントによる強化学習を行うための環境。
  * GitHubページ上ではnotebookによるチュートリアルも提供されている。
  * 学習後にエージェントを動かすためのDockerコンテナを公式サイトにpushすると、評価スクリプトを流してランキングしてくれるよう
* [TorchCraft](https://github.com/TorchCraft/TorchCraft)
  * StarCraftを学習するためのフレームワーク
  * なぜかalibabaからPyTorchと連携できるようにするためのツールが公開されている([torchcraft-py](https://github.com/alibaba/torchcraft-py))
  * 強化学習用エージェントを作りやすくした[TorchCraftAI](https://github.com/TorchCraft/TorchCraftAI)も公開された。
* [StarCraft I (BroodWar) docker images](https://github.com/Games-and-Simulations/sc-docker)
  * StarCraftを利用した学習などを行いやすくするために、環境をDocker化したもの。
  * Dockerの中にStarCraftとWindowsアプリケーションを動かすためのWineHQ、StarCraftを操作するためのBWAPIがまとめられており、Host側で面倒なセットアップを行うことなく操作ができる。
* [pysc2](https://github.com/deepmind/pysc2)
  * DeepMindの作成した、StarCraft IIで学習するための環境。
* [MAgent](https://github.com/geek-ai/MAgent)
  * 複数、数百～数千のエージェントによる強化学習のシミュレーションに利用可能な環境。
  * 動作確認はLinuxのみだが、Windows/Macでも動かせるよう。
* [trfl](https://github.com/deepmind/trfl/)
  * TensorFlowで強化学習を実装するためのモジュール集ともいえるライブラリ。
  * 様々なアルゴリズムで共通して利用されるアルゴリズムのパート(Q-learningやPolicy Gradientなど)の実装が提供されている。deepmind/trflこれを組み立てれば、実装ミスを最小限に抑えることができる。
* [Horizon](https://github.com/facebookresearch/Horizon)
  * Facebookが公開した強化学習を行うためのフレームワーク。
  * プロトタイピングというより実用のための開発が意識されており、シミュレーターが遅いケースのために事前にとっておいた状態情報を使うなどの機能がある。
  * [White paper](https://research.fb.com/publications/horizon-facebooks-open-source-applied-reinforcement-learning-platform/)にはFacebookの通知で使用された実績が書かれている。
* [SenseAct](https://github.com/kindredresearch/SenseAct)
  * OpenAI Gym的なシミュレーション環境を、物理の世界へ拡張しようという試み。
  * 具体的には、一般に販売されているロボットを使ってタスクを定義し、それをOpenAI Gymライクなインタフェースで操作できるようにしている。 
* [Nevergrad: An open source tool for derivative-free optimization](https://github.com/facebookresearch/nevergrad)
  * 進化戦略のような勾配を使わない最適化を行うライブラリ。ハイパーパラメーターサーチに使うことを想定しているよう。
* [CoinRun](https://github.com/openai/coinrun)
  * 強化学習アルゴリズムの転移性能を検証するための環境を公開。OpenAIで以前行われたソニックのコンテストをベースとしており、学習時とは異なる難易度で評価が行えるようになっている(難易度は1~3で、難易度に応じてコースが生成される)。過学習を検出するのにも使える。
* [Neural MMO](https://github.com/openai/neural-mmo)
  * 長時間、しかも多数のエージェントによる学習を行うための環境。
  * 比較的短いサイクルでリセットされる既存の環境と異なり長時間続くため、提供されている標準実装(素のPolicy Gradient)では生存時間を報酬としている。
* [streetlearn](https://github.com/deepmind/streetlearn)
  * 強化学習でナビゲーションを学習する環境。Google Street Viewをベースにしており、実世界に近い環境となっている。
* [purdue-biorobotics/flappy](https://github.com/purdue-biorobotics/flappy)
  * 昆虫やハチドリのような、細かい羽ばたきによる飛行を学習するシミュレーター。空中を飛ぶ小型のロボットには不可欠だが、羽ばたき(=Action)が多い一方安定的な飛行を要求される。
  * 実際にロボットを製作し、その挙動を再現している。
* [PHYRE: A new AI benchmark for physical reasoning](https://phyre.ai/)
  * 物理法則を利用して目的を達成するタスクを集めた環境。
  * タスクでは下にあるカップへボールを入れるために、坂を傾けてボールを転がす、といった行動をとることが求められる。タスクは基本一手~数手でクリア可能で、物理法則に関する事前知識が求められる内容になっている。
* [Spriteworld](https://github.com/deepmind/spriteworld)
  * 複数オブジェクトを操作する環境を簡単に作成できるツール。
  * 「四角をセンターに」「青色をまとめる」といった、色や形状に基づくゴールを設定した環境を作成することができる。

# Others

* [dgl](https://github.com/dmlc/dgl/)
  * Graph Convolution/Neural Networkを簡単に使えるライブラリ。主要なアルゴリズムがサポートされている。PyTorch/MXNetバックエンドで作成されており、パフォーマンスも意識されている。

## Annotation Tool

* [visdial-amt-chat](https://github.com/batra-mlp-lab/visdial-amt-chat)
  * 画像に関する質問＋回答のデータを収集するためのアプリケーション。[実際のアプリケーション開発](https://visualdialog.org/)に使用したものを公開してくれている
* [doccano](https://github.com/chakki-works/doccano)
  * 自然言語処理のためのアノテーションツール。テキスト分類、系列ラベリング、系列変換(翻訳)という3つのタスクに対応している。

## DataAPI

* [pandas](http://pandas.pydata.org/)
  * 表形式データを扱いやすくるためのツール。描画機能も付属しているため、簡単なグラフならpandasのみでかなりいける([こちら](http://pbpython.com/effective-matplotlib.html)参照)。
* [feather](https://github.com/wesm/feather)
  * 様々なプラットフォーム(Python/pandas、Rなど)で共通して扱え、なおかつ高速性を担保するデータフレームのファイルフォーマット、およびそれを扱うツール
  * いったんfeather形式のファイルフォーマットで保存し、それを読み込む形になる。この操作は特にメモリに乗らないような大規模なデータセットで力を発揮し、デモでは600MBのファイルを1秒たらずで読み込めている([Feather: A Fast On-Disk Format for Data Frames for R and Python, powered by Apache Arrow](https://blog.rstudio.org/2016/03/29/feather/))。

## Visualization

* [bokeh](http://bokeh.pydata.org/en/latest/)
  * Pythonのグラフ描画ツールで、描画結果からHTMLコンテンツの作成が可能。このため、Jupyter上で使うほかWebページに埋め込んだりすることもできる
* [seaborn](https://seaborn.pydata.org/)
  * Matplotlib上で稼働し、より使いやすく、きれいに描画できる
* [HyperTools](http://hypertools.readthedocs.io/en/latest/)
  * 高次元のデータの可視化に特化したツール
* [Picasso](https://medium.com/merantix/picasso-a-free-open-source-visualizer-for-cnns-d8ed3a35cfc5)
  * CNNの層を可視化するためのツール。Keras/TensorFlowのチェックポイントファイルをベースに動く。
* [scattertext](https://github.com/JasonKessler/scattertext)
  * コーパス内の単語を使って散布図を作れるビジュアライゼーションツール。どの分類にどの単語が効いているかなどを、視覚的にみることができる

## Experiment Support

* [Data Version Control Tutorial](https://blog.dataversioncontrol.com/data-version-control-tutorial-9146715eda46)
  * データとソースコードを紐付けてバージョン管理を行うことのできるツール

## Machine Learning PipeLine

Auto MLについては、こちらの記事を参照。今後は、まずは基本的な前処理/可視化を終えた後は一旦自動化パイプラインに流して見るというのが基本になると考えられる。

[その機械学習プロセス、自動化できませんか？](https://qiita.com/Hironsan/items/30fe09c85da8a28ebd63)

* [featuretools](https://github.com/Featuretools/featuretools)
  * データを渡すと、平均/最大/最小などをはじめとした各種統計量の特徴量を自動的に生成してくれるツール。

## Benchimark Implementation

ベースラインとして利用できるような実装を紹介します。

* [OpenAI Baselines](https://github.com/openai/baselines)
* [Memory-Augmented Neural Networks](https://github.com/facebook/MemNN)
* [Neural-Dialogue-Generation](https://github.com/jiweil/Neural-Dialogue-Generation)
* [PixelCNN++](https://github.com/openai/pixel-cnn)
* [benchmark_results](https://github.com/foolwood/benchmark_results)
  * Visual Trackingのベンチマーク集
* [chainerrl](https://github.com/chainer/chainerrl)
  * Chainerで実装された強化学習のアルゴリズム集
* [keras-rl](https://github.com/matthiasplappert/keras-rl)
  * Kerasで実装された強化学習のアルゴリズム集
* [Molecular Sets (MOSES)](https://github.com/molecularsets/moses)
  * 創薬に使用されている代表的なアルゴリズムを実装したリポジトリ。
