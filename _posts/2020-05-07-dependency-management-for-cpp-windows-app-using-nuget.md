---
layout: post
title: "C++ 기반 윈도우 데스크톱 애플리케이션 의존성 관리"
date: 2020-05-07 23:00:00 +0900
author: gigone_lee
categories: [Development]
tags: [c++, nuget]
comments: true
name: dependency-management-for-cpp-windows-app-using-nuget
image: boost_in_nuget.png
---

라이브러리를 사용할 때는 프로젝트가 사용하는 라이브러리 목록과 버전을 명확히 관리해야 합니다. 개발 환경과 배포 환경의 라이브러리 버전 차이가 의도치 않은 동작을 만들 수 있기 때문입니다. 또한 모든 팀원이 같은 라이브러리 버전을 사용해야 팀원 개개인이 동일한 동작을 기대하고 기능을 개발할 수 있습니다.

C++처럼 개발 플랫폼(운영체제, 컴파일러 등)이 다양한 환경에서는 목록과 버전 외에 확인해야 할 게 더 많습니다. 예를 들면 라이브러리의 아키텍처(x32, x64, arm)와 컴파일러 버전(vc16, gcc 등)과 컴파일러 옵션, CPU 명령어 셋 집합이 배포 환경에서 지원하는지 등을 확인해야 합니다.

대부분의 개발 환경은 손쉽게 라이브러리 목록과 버전을 관리할 수 있습니다. 파이썬을 예로 들면, requirements.txt 파일을 통해 프로젝트가 사용하는 라이브러리와 버전을 명확하게 관리할 수 있습니다.

{% include image.html name=page.name file="python_requirement_txt.png" description="Python 기반 프로젝트의 라이브러리 의존성 관리 파일(requirements.txt)" class="full-image" %}

그러나 모든 개발 환경이 이런 편리한 기능을 제공하지는 않습니다. 저희 코어 팀에서 사용하는 C++도 마찬가지였습니다. 제가 팀에 합류한 시점에는 라이브러리들을 모두 별도 디렉토리에 넣고 환경 변수로 관리했는데, 이는 개발 팀원 간 통일된 개발 환경을 만들기 어렵게 했습니다. 구체적으로는 다음과 같은 사례가 있습니다.
-	버그를 찾기 위해 환경 변수를 바꿔 라이브러리 버전을 변경했을 때, 이를 다시 직접 바꾸지 않으면 계속 변경된 라이브러리로 소프트웨어를 개발하게 됩니다. 이를 강제로 되돌릴 방법도 없고, 버그가 생기거나 빌드가 되지 않을 때까지 알아채기도 쉽지 않습니다.
-	개발 환경을 설정하기가 굉장히 까다로웠습니다. 소프트웨어를 빌드하기 위해 설정해야 할 환경 변수가 많고, 의존 관계가 명확하지 않다 보니 라이브러리가 누락되거나 파일 한 개가 바뀌었을 때 이를 발견하기가 쉽지 않았습니다.

이러한 문제들을 해결하기 위해서는 라이브러리의 버전과 의존 관계를 명확히 해야 하고 모든 개발자들이 통일된 개발 환경에서 제품을 만들 수 있게 강제하는 시스템이 필요했습니다. 이 문제를 해결할 방법을 찾던 중, 저희는 Nuget 패키지를 활용하기로 결정했습니다. 이번 글에서는 간단하게 Nuget 패키지를 만드는 방법과 저희 팀에서 Nuget을 도입하면서 얻게 된 장점을 함께 설명하겠습니다.

## Nuget 저장소?

Nuget은 파이썬의 PIP나 자바스크립트의 NPM처럼 .NET 소프트웨어의 라이브러리를 쉽게 사용할 수 있게 만든 패키지 저장소입니다. C++를 직접적으로 지원하는 건 아니지만 Visual Studio IDE와 연동되는 장점이 있기 때문에 몇 가지만 설정하면 충분히 C++ 라이브러리 저장소로도 사용이 가능합니다. Boost와 같은 범용적인 라이브러리는 이미 Nuget 공개 저장소에 올라왔을 정도로, 많은 사람들이 사용하고 있기도 합니다.

{% include image.html name=page.name file="boost_in_nuget.png" description="Nuget 저장소에 있는 boost 라이브러리" class="full-image" %}

## Nuget C++ 라이브러리 만들기

라이브러리를 만드는 작업은 그리 어렵지 않습니다. OpenCV를 예로 들어 설명하겠습니다. 먼저 패키지 정의(.nuspec) 파일을 생성한 다음, OpenCV 빌드 결과물을 다음과 같이 형식에 맞게 지정하면 됩니다(파일 안에 있는 각각의 항목에 대한 자세한 설명은 <a href="https://docs.microsoft.com/ko-kr/nuget/reference/nuspec" target="_blank">여기</a>를 참고해주세요).

```xml
<?xml version="1.0"?> 
<package xmlns="http://schemas.microsoft.com/packaging/2010/07/nuspec.xsd"> 
  <metadata> 
    <id>opencv-vc14-win64</id> 
    <version>3.4.1</version> 
    <authors>SuaKIT Development Team</authors>
    <owners>SuaKIT</owners>
    <requireLicenseAcceptance>false</requireLicenseAcceptance> 
    <description>Canonical OpenCV library</description> 
   </metadata>
   <files>
    <file src="opencv-3.4.1-vc14-x64\x64\vc14\bin\*.dll" target="lib" />
    <file src="opencv-3.4.1-vc14-x64\x64\vc14\lib\*" target="lib" />
    <file src="opencv-3.4.1-vc14-x64\x64\vc14\bin\*.exe" target="bin" />
    <file src="opencv-3.4.1-vc14-x64\include\**" target="include" />
    <file src="opencv-3.4.1-vc14-x64\etc\**" target="etc" />
    <file src="opencv-3.4.1-vc14-x64\LICENSE" />
    <file src="opencv-3.4.1-vc14-x64\OpenCVConfig.cmake" />
    <file src="opencv-3.4.1-vc14-x64\OpenCVConfig-version.cmake" />
  </files>
</package>
```

작성한 nuspec 파일을 사용 가능한 패키지(nupkg) 파일로 만드는 작업도 간단합니다. <a href="https://www.nuget.org/downloads" target="_blank">Nuget 홈페이지</a>에 있는 Nuget 바이너리 파일 하나만 받아, 다음과 같이 파워셸이나 명령 프롬프트에서 실행하면 파일이 생성됩니다.

```cmd
nuget.exe pack opencv-vc14-win64.nuspec
```

## 종속성 관리
때로는 라이브러리를 사용하기 위해 다른 라이브러리의 설치를 강제할 필요가 있습니다. Nuget을 이용하면 이 의존관계를 쉽게 표현할 수 있는데, 예를 들어 앞서 작성했던 nuspec 파일에 다음 내용을 추가하면
```xml
    <description>Canonical OpenCV library</description> 
    <dependencies>
      <dependency id="boost-vc16-win64static" version="1.71.0" />
    </dependencies>
   </metadata>
```
저장소에서 이 라이브러리를 설치하는 것 만으로 boost 라이브러리가 함께 설치됩니다.

{% include image.html name=page.name file="dependency_example.png" description="설정된 boost 라이브러리 종속성" class="full-image" %}

## Nuget 라이브러리 배포하기

이제 라이브러리 파일을 배포해야 합니다. 저희가 라이브러리로 사용하는 파일 중 일부는 외부에 공개할 수 없었기 때문에 사내 저장소를 만들기로 했습니다.

일반적으로 사설 저장소를 만들려면 많은 환경 설정과 서버가 필요한데, Nuget은 사설 저장소를 구축하는 비용 또한 저렴합니다. 그 이유는 디렉토리 기반 저장소를 지원하기 때문입니다.

다음 그림을 보면, 로컬 디렉토리가 사설 저장소로 설정된 것을 볼 수 있습니다. 

{% include image.html name=page.name file="set_local_directory_as_private_repository.png" description="로컬 디렉토리를 사설 저장소로 설정" class="full-image" %}

저희 개발 팀은 개발 팀에서 사용하는 네트워크 스토리지(NAS)를 사설 저장소로 사용하기로 하고, 네트워크 스토리지에 의존 관계에 있는 모든 라이브러리를 넣었습니다. 지금은 보다 많은 사람들이 이 라이브러리를 사용할 수 있게 사내 웹 서버에 올려 두었습니다.

## Nuget 라이브러리 사용 시 주의해야 할 점

이렇게 라이브러리의 버전을 명시하고 의존 관계를 만든 후, 저장소를 만드는 것 까지는 좋았으나, Nuget 저장소 자체가 C++에 특화된 것이 아니기 때문에 해결해야 할 문제가 있었습니다.

처음에는 컴파일에 사용할 include 파일을 찾을 수 없는 문제가 생겼습니다. 이 문제는 추가 포함 디렉토리에 패키지 경로를 지정하면 해결이 가능했습니다.

{% include image.html name=page.name file="add_path_of_package.png" description="추가 포함 디렉토리에 지정된 패키지 경로" class="full-image" %}

두 번째 문제는, 런타임 시 사용할 DLL이 빌드 결과물에 포함되지 않는다는 것이었습니다. 그래서 컴파일이 되도 실행은 되지 않는 문제가 발생했습니다. 그러나 이 문제 또한 빌드 이벤트에 DLL을 복사하게 설정하는 것으로 쉽게 해결하였습니다.

{% include image.html name=page.name file="add_build_event.png" description="빌드 이벤트에 DLL 복사를 설정" class="full-image" %}

마지막으로 해결해야 할 문제는 컴파일 환경입니다. 앞서 이야기한 것처럼 C++ 언어는 아키텍처, 지원 명령어 셋, 라이브러리 컴파일 옵션 등 함께 관리해야 할 여러 요소들이 많습니다. 이 모든 것들을 Nuget으로 해결할 수는 없습니다. 저희 팀은 이를 보완하기 위해, 패키지 이름에 다음 내용을 반드시 명시하는 관습을 따르고 있습니다.
-	아키텍처(x64)와 빌드 환경(vc15, vc16 등)
-	사용 시 주의해야 할 컴파일 옵션 (static 빌드, 특정 옵션 제거 등)

## 마무리

이렇게 필요한 라이브러리들을 모두 옮기니, 저희 팀은 다음과 같은 문제들을 해결할 수 있었습니다.
-	프로젝트 빌드에 필요한 라이브러리는 모두 Nuget 저장소에 있고, 저장소 경로와 Include, Linking 경로도 모두 프로젝트에 포함되어 있습니다. 빌드 버튼을 누르면 자동으로 저장소에서 패키지를 받고 의존성 목록에 따라 설치합니다. 결과적으로는 소스코드만 받아도 쉽게 빌드 환경을 구성할 수 있습니다. 
-	제품 개발에 참여하는 모든 개발자가 프로젝트와 Nuget 저장소에 의해 고정된 라이브러리 파일을 사용하기 때문에 같은 개발 환경을 보장할 수 있습니다. 
-	라이브러리 버전을 보다 쉽게 확인할 수 있게 됐습니다. 저희 팀은 GIT을 사용하고 있고, 프로젝트 파일에 포함된 Include와 Linking 경로들 또한 git에 의해 추적되기 때문에 커밋하기 전에 사용중인 라이브러리 상태를 다시 한번 확인하고 검증할 수 있습니다.

개발 팀에서 윈도우 기반의 C++ 개발 환경을 구축하는 일은 쉽지 않습니다. 리눅스의 경우 APT, YUM/DNF와 같은 저장소를 사용할 수 있지만, 윈도우는 아직까지 표준이라고 할 패키징 관리 시스템이 없기 때문입니다. 그러나 Nuget을 이용하면 적은 리소스로 리눅스 못지않게 라이브러리의 목록과 버전을 관리할 수 있습니다.
