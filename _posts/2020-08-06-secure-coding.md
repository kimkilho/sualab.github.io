---
layout: post
title: "안전한 C++ 코드 만들기"
date: 2020-08-06 18:00:00 +0900
author: gigone_lee
categories: [Development]
tags: [c++, modern c++, convention]
comments: true
name: secure-coding
image: thumbnail_secure_coding.jpg
---

C++만큼 개발자에게 다양한 선택지를 주는 언어는 거의 없습니다. C++의 언어적 특징은 제품의 형태에 최적화된 형태로 코드를 만드는 데 많은 도움을 주지만, 개발자가 신경 써야 할 영역이 많기 때문에 개발자의 역량에 따라 제품의 완성도 편차가 심하며 안전하지 않은 코드를 작성하기 쉬운 환경에 노출된다는 단점이 있습니다.


안전한 코딩은 단순히 잘 동작한다는 것 이상을 뜻합니다. 가독성이 좋아 유지보수가 쉽고, 잘 사용하기는 쉽지만 잘못 쓰기는 어려운 코드를 만드는 것이 `안전한 코딩(Secure coding)`의 본질이라 생각합니다.

그럼 코그넥스 개발팀에서 사용 중인 C++ 문법과 디자인 패턴에는 어떤 것들이 있고, 이러한 것들이 어떻게 안전한 코드를 만드는지 살펴보겠습니다.


## 스마트 포인터 (std::shared_ptr과 std::unique_ptr)

저희 제품에서는 특정 핵심 로직을 제외한 모든 곳에서 new 할당자 대신 스마트 포인터를 사용합니다.

`std::shared_ptr<T>`와 `std::unique_ptr<T>`는 객체가 해제되는 시점에서 자동으로 할당된 객체를 파괴하는 기능을 제공하며 객체 복사가 가능한 std::shared_ptr와 복사가 불가능하고 이동만 가능한 std::unique_ptr 모두 사용하고 있습니다.

두 객체 모두 함수 안에서 임시로 사용되는 객체나 변수가 아닌, 여러 메서드나 클래스에서 오랫동안 유지해야 하는 객체나 변수에 사용하는 것이 좋습니다.

>    객체를 반드시 복사해야 할 필요가 없다면 항상 unique_ptr을 선호하는 것이 안전합니다. 그 이유는 shared_ptr 객체는 복사된 객체를 가지고 있는 모든 곳에서 해제하기 전까지 객체 해제가 되지 않아 메모리 누수(memory leak)을 만들기 상대적으로 더 쉽기 때문입니다. unique_ptr은 객체 복사가 불가능하기 때문에 안전하게 필요한 곳에서만 사용하고 필요한 시기에 해제하기가 쉽습니다.
>
>    반대로 깊은 복사(deep-copy)가 잦을 때는 포인터 자체를 사용하지 않는 것도 좋은 방법입니다.

스마트 포인터를 사용할 때는 반드시 `make_shared / make_unique` 템플릿을 이용하는 것이 좋은데, 그 이유는 크게 두 가지가 있으며 [허브 서터의 이야기](https://herbsutter.com/2013/05/29/gotw-89-solution-smart-pointers/)를 인용하면 다음과 같습니다.

먼저 new 할당자를 사용할 때 어떤 일이 벌어지는지 살펴봅시다.

```c++
auto sp1 = shared_ptr<widget>{ new widget{} };
auto sp2 = sp1;
```

[그림 1] sp1과 sp2가 할당되는 과정

{% include image.html name=page.name file="p1.png" description="sp1과 sp2가 할당되는 과정" class="full-image" %}

처음 Widget이 new 할당자를 통해 생성되면, shared_ptr 객체가 한번더 생성되면서 Widget을 가리킵니다. 총 2번의 객체 할당이 발생하게 됩니다. 그렇다면 make_shared는 어떨까요?

```c++
auto sp1 = make_shared<widget>();
auto sp2 = sp1;
```

[그림 2] sp1과 sp2가 할당되는 과정

{% include image.html name=page.name file="p2.png" description="sp1과 sp2가 할당되는 과정" class="full-image" %}

make_shared 템플릿을 사용하면, Widget을 클래스 인자로 받는 shared_ptr 생성 1회로 끝납니다. new 할당자와 다르게 총 1번의 객체 할당만 발생하게 됩니다.


그러나 make_shared를 사용해야만 하는 더 큰 이유는, new 할당자를 사용할 때 `메모리 누수(memory leak)`가 발생할 수 있기 때문입니다.

```c++
void sink( unique_ptr<widget>, unique_ptr<gadget> );

sink( unique_ptr<widget>{new widget{}},
        unique_ptr<gadget>{new gadget{}} );
```

위 코드에서 new gadget{} 안에서 예외가 발생할 경우 new widget{}은 해제되지 않아 메모리 누수가 발생할 수 있습니다. 또한 함수를 호출할 때 인자의 평가(Evaluation)는 `명시되지 않은 행동(Unspecified Behavior)`으로 컴파일러마다 정의되는 행동이 다르기 때문에, new widget{} 안에서 예외가 발생할 때 new gadget{}으로 생성된 객체가 해제되지 않는 반대의 상황 또한 발생할 수 있습니다.

이러한 이유로 항상 make_shared 또는 make_unique를 사용하는 것이 안전합니다.

## static_cast<T>, dynamic_cast<T>, 그리고 reinterpret_cast<T>

캐스팅 연산은 보통 3가지(const_cast 제외) 중 1개를 선택하게 되는 데, static_cast와 dynamic_cast, 그리고 reinterpret_cast 캐스팅이 있습니다.


static_cast 연산은 컴파일 시점에서 형 변환을 하며 대부분의 int, double과 같은 원시 타입(primitive type) 값을 변환할 때 사용하거나, 안전하다고 확신할 수 있는 때에(null이 아닌 경우)만 부모 클래스로 형 변환 등을 할 때 사용합니다. 

```c++
int a = 42;
double b = static_cast<double>(a);
```

dynamic_cast 연산은 컴파일 시점이 아닌 런타임 시점에서 형 변환을 합니다. 형변환이 실패한 경우 null을 반환하기 때문에 안전하다고 확신할 수 없는 모든 객체의 형 변환에 사용합니다.

```c++
Class A {};
Class B : public A {};
B *b = new B();
A *a = dynamic_cast<A *>(b);
assert(a); // a는 null이 아닙니다. 형변환이 실패하면 null 여부를 검사해 확인할 수 있습니다.
```

마지막으로 reinterpret_cast 연산은 가장 위험한 연산으로, C 스타일의 캐스팅과 동일한 형 변환을 수행합니다. 포인터를 정수로 저장하거나 정수를 포인터 주소로 사용하는 경우, 또는 void * 포인터 값을 임의의 클래스 포인터 주소로 사용하는 경우를 예로 들 수 있습니다. 

> reinterpret_cast 연산은 형 변환에 사용되는 두 객체 값 제어가 가능한 경우가 아니면 사용하지 않는 게 좋습니다.

## assert 와 static_assert, 그리고 enable_if

다음으로 볼 내용은 assert와 static_assert 입니다. 그리고 가능한 경우 함께 사용할 수 있는 std::enable_if도 소개하려 합니다.

먼저 assert(eval)는 디버그 모드에서만 동작하는 메서드로 eval 식이 참이 아닌 경우 즉시 프로그램을 강제 종료하는 기능을 가지고 있습니다. 

assert는 로직이 런타임에서 의도한대로 동작하는 지 확인하고 싶을 때 사용하면 좋습니다. 검사 식 자체가 로직의 방향을 이해하는 데 큰 도움이 되기 때문에 복잡한 코드의 가독성을 크게 끌어올리는 장점을 가지고 있습니다.

```c++
...
// 값을 찾았을 때는 level 값이 반드시 100 이상이고, 그렇지 않으면 100 미만이어야 함을 보장하는 assert
// 이 식을 가정하에 코드가 작성됐다는 걸 알 수 있기 때문에 더 빠르게 코드를 이해할 수 있습니다.
assert((found && level >= 100) || (!found && level < 100) );
```

static_assert는 런타임이 아닌 컴파일 시점에서 의도한대로 동작하는 지 확인할 때 사용하며, 전처리 매크로가 의도한 대로 설정됐는지, 템플릿 인자에 의도한 타입이 들어왔는지 확인할 때 유용하게 사용할 수 있습니다.

```c++
// 템플릿 인자 T의 타입이 bool이 아닌 경우 컴파일 에러가 발생합니다.
static_assert(std::is_same<decltype(T), bool>::value, "T must be bool");
```

그러나 템플릿 인자 타입을 검사할 때는 std::enable_if를 사용하는 게 조금 더 좋습니다. 그 이유는 static_assert의 경우 생성된 코드 안에서 에러가 출력되고 에러가 발생한 위치를 알려주지 않는 반면, std::enable_if는 해당하는 템플릿 함수가 없다는 에러가 출력되고 템플릿 함수를 호출한 쪽에서 에러가 발생하기 때문에 원인을 쉽게 찾을 수 있기 때문입니다.

```c++
// double 또는 int를 인자로 사용하는 CheckValue() 템플릿 함수만 사용 가능합니다.
template <typename T,
        std::enable_if<
        std::is_same<T, std::double>::value ||
        std::is_same<T, std::int>::value>::value> * = nullptr>
void CheckValue(const T& value);
```

## const reference 키워드

const reference 는 const & 를 뜻하며 변경이 불가능하고 참조만 가능한 변수나 인자를 뜻합니다. 이 키워드는 함수의 인자, 스택 로컬 변수 등을 선언할 때 사용하는 게 좋습니다. 

```c++
// 사용 예 1
void func(const std::string &str)
{
    // 사용 예 2
    const std::string &name = GetStringValue()
}
```

const reference는 일반적인 reference (&)와는 조금 다른 특징을 가지고 있습니다. 첫 번째는 const 키워드로 인해 변경이 불가능하다는 것이고, 두 번째는 reference임에도 객체의 `생명 주기(lifetime)`가 더 길다는 것입니다.

위 예제의 `사용 예 2`를 보면 마치 GetStringValue() 가 반환하는 std::string 객체를 복사하지 않고 참조만 하는 것처럼 보이지만, 실제로는 const reference에 의해 생명 주기가 연장되어 name 변수가 사라질 때까지 유지됩니다(복사되는 것이 아니며 RVO와도 아무런 관계가 없다는 점에 주의하세요).

const reference는 불필요한 복사를 막아주고 실수로 변수를 수정하는 일을 쉽게 제한할 수 있다는 점에서 안전한 코드를 만드는 데 큰 도움을 줄 수 있습니다.


## 요약

이번 글을 통해 소개드린 내용 말고도 많은 장치들을 활용하여 안전하게 코드를 만들 수 있습니다. 특히 템플릿이나 람다 식은 디버깅을 어렵게 만들 수 있기 때문에 가능한 적게 사용한다거나, constexpr if 등을 이용해 중복 코드를 최대한 제거하여 실수할 여지를 줄이는 것 등이 있을 것입니다. 때로는 스크립트로 생성되는 인터페이스로만 코드 사용이 가능하게 하는 것도 좋은 방법입니다. 

> 물론 때로는 최적화를 위해 이해하기 어려운 코드를 만들어야 할 때도 있습니다. 이 때는 핵심적인 부분은 최대한 숨기고 주석으로 대체하되, 성능에 별 지장이 없는 논리적인 부분을 잘못 사용하기 어렵게 만드는 것이 매우 중요합니다.


핵심은 최대한 일관성 있는 코드를 만들고, 실수하기 어려운 환경을 만드는 것입니다. 저는 그것이 안전한 코드를 만드는 데 필요한 가장 중요한 원칙이라 생각합니다.