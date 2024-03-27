#include "catch_amalgamated.hpp"
#include "Voigt.hpp"

using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

using namespace std::complex_literals;
constexpr int n_a = 11;
constexpr f64 a[] = {0. ,  1.5,  3. ,  4.5,  6. ,  7.5,  9. , 10.5, 12. , 13.5, 15.};
constexpr int n_v = 11;
constexpr f64 v[] = {-600., -480., -360., -240., -120.,    0.,  120.,  240.,  360., 480.,  600.};
// NOTE(cmo): Table generated using standard Faddeeva package
std::complex<f64> expected_voigt[n_a][n_v] = {{0.0 + -0.0009403172785794419i, 0.0 + -0.0011753975165114238i, 0.0 + -0.0015671993339730376i, 0.0 + -0.002350810338143017i, 0.0 + -0.004701743129206853i, 1.0 + 0.0i, 0.0 + 0.004701743129206853i, 0.0 + 0.002350810338143017i, 0.0 + 0.0015671993339730376i, 0.0 + 0.0011753975165114238i, 0.0 + 0.0009403172785794419i},
{2.350785033925748e-06 + -0.0009403114015923698i, 3.673097310984479e-06 + -0.0011753860380075971i, 6.529934241665077e-06 + -0.001567172125598784i, 1.469224575298719e-05 + -0.0023507185092156626i, 5.876668667091898e-05 + -0.004701008469093714i, 0.3215854164543176 + 0.0i, 5.876668667091898e-05 + 0.004701008469093714i, 1.469224575298719e-05 + 0.0023507185092156626i, 6.529934241665077e-06 + 0.001567172125598784i, 3.673097310984479e-06 + 0.0011753860380075971i, 2.350785033925748e-06 + 0.0009403114015923698i},
{4.7014819147595175e-06 + -0.0009402937710719212i, 7.345979406560983e-06 + -0.0011753516038412275i, 1.3059188310712707e-05 + -0.001567090506144239i, 2.9381048339668743e-05 + -0.0023504430654750436i, 0.00011747830061245449 + -0.004698805865811581i, 0.17900115118138998 + 0.0i, 0.00011747830061245449 + 0.004698805865811581i, 2.9381048339668743e-05 + 0.0023504430654750436i, 1.3059188310712707e-05 + 0.001567090506144239i, 7.345979406560983e-06 + 0.0011753516038412275i, 4.7014819147595175e-06 + 0.0009402937710719212i},
{7.052002500428018e-06 + -0.0009402643883403197i, 1.1018431113353106e-05 + -0.0011752942180472538i, 1.9587082270669617e-05 + -0.0015669544926110936i, 4.4062967283024e-05 + -0.0023499841359950744i, 0.0001760799410213542 + -0.004695139444083503i, 0.12248480427384142 + 0.0i, 0.0001760799410213542 + 0.004695139444083503i, 4.4062967283024e-05 + 0.0023499841359950744i, 1.9587082270669617e-05 + 0.0015669544926110936i, 1.1018431113353106e-05 + 0.0011752942180472538i, 7.052002500428018e-06 + 0.0009402643883403197i},
{9.402258670892436e-06 + -0.0009402232556009945i, 1.4690237342030347e-05 + -0.0011752138873492603i, 2.611293665717871e-05 + -0.0015667641133256635i, 5.8734567480192303e-05 + -0.002349341935730964i, 0.00023451704982059145 + -0.004690016057008226i, 0.09277656780053836 + 0.0i, 0.00023451704982059145 + 0.004690016057008226i, 5.8734567480192303e-05 + 0.002349341935730964i, 2.611293665717871e-05 + 0.0015667641133256635i, 1.4690237342030347e-05 + 0.0011752138873492603i, 9.402258670892436e-06 + 0.0009402232556009945i},
{1.1752162339158445e-05 + -0.0009401703759381684i, 1.8361183129287965e-05 + -0.001175110621157508i, 3.2636072713627034e-05 + -0.0015665194079241438i, 7.339242187854064e-05 + -0.0023485167652676324i, 0.0002927355802200154 + -0.0046834452540970165i, 0.07457369306287667 + 0.0i, 0.0002927355802200154 + 0.0046834452540970165i, 7.339242187854064e-05 + 0.0023485167652676324i, 3.2636072713627034e-05 + 0.0015665194079241438i, 1.8361183129287965e-05 + 0.001175110621157508i, 1.1752162339158445e-05 + 0.0009401703759381684i},
{1.4101625462277544e-05 + -0.0009401057533162796i, 2.2031053679775194e-05 + -0.00117498443156618i, 3.9155812626258094e-05 + -0.0015662204273319863i, 8.803311413855084e-05 + -0.0023475090104682597i, 0.0003506821619491842 + -0.004675439236915043i, 0.06230772403777468 + 0.0i, 0.0003506821619491842 + 0.004675439236915043i, 8.803311413855084e-05 + 0.0023475090104682597i, 3.9155812626258094e-05 + 0.0015662204273319863i, 2.2031053679775194e-05 + 0.00117498443156618i, 1.4101625462277544e-05 + 0.0009401057533162796i},
{1.6450560052336846e-05 + -0.0009400293925792391i, 2.569963440795462e-05 + -0.0011748353333498386i, 4.5671479758597775e-05 + -0.0015658672337374175i, 0.00010265324127919102 + -0.002346319142023741i, 0.0004083042621110336 + -0.004666012802710621i, 0.05349189974656411 + 0.0i, 0.0004083042621110336 + 0.004666012802710621i, 0.00010265324127919102 + 0.002346319142023741i, 4.5671479758597775e-05 + 0.0015658672337374175i, 2.569963440795462e-05 + 0.0011748353333498386i, 1.6450560052336846e-05 + 0.0009400293925792391i},
{1.8798878187434405e-05 + -0.0009399412994495223i, 2.936671097987561e-05 + -0.001174663343959107i, 5.21823988850237e-05 + -0.0015654599005591257i, 0.00011724941630148367 + -0.0023449477149040205i, 0.0004655503406534079 + -0.004655183276515927i, 0.04685422101489376 + 0.0i, 0.0004655503406534079 + 0.004655183276515927i, 0.00011724941630148367 + 0.0023449477149040205i, 5.21823988850237e-05 + 0.0015654599005591257i, 2.936671097987561e-05 + 0.001174663343959107i, 1.8798878187434405e-05 + 0.0009399412994495223i},
{2.11464920226373e-05 + -0.0009398414805270984i, 3.3032069354844735e-05 + -0.0011744684835155617i, 5.868789642330936e-05 + -0.0015649985124081471i, 0.00013181827078604546 + -0.002343395367712504i, 0.0005223699995144099 + -0.00464297043229652i, 0.04167809676408815 + 0.0i, 0.0005223699995144099 + 0.00464297043229652i, 0.00013181827078604546 + 0.002343395367712504i, 5.868789642330936e-05 + 0.0015649985124081471i, 3.3032069354844735e-05 + 0.0011744684835155617i, 2.11464920226373e-05 + 0.0009398414805270984i},
{2.3493313800919454e-05 + -0.000939729943288194i, 3.669549582697624e-05 + -0.0011742507748058557i, 6.518730066597498e-05 + -0.001564483165043988i, 0.0001463564574604445 + -0.0023416628219449375i, 0.0005787141245660081 + -0.0046293964038128495i, 0.03752960638850577 + 0.0i, 0.0005787141245660081 + 0.0046293964038128495i, 0.0001463564574604445 + 0.0023416628219449375i, 6.518730066597498e-05 + 0.001564483165043988i, 3.669549582697624e-05 + 0.0011742507748058557i, 2.3493313800919454e-05 + 0.000939729943288194i}
};

TEST_CASE("Test Humlicek Impl", "[test_humlicek]") {
    for (int ia = 0; ia < n_a; ++ia) {
        for (int iv = 0; iv < n_v; ++iv) {
            std::complex<fp_t> computed = humlicek_voigt(fp_t(a[ia]), fp_t(v[iv]));

            REQUIRE_THAT(computed.real(), WithinRel(fp_t(expected_voigt[ia][iv].real()), FP(1e-3)));
            REQUIRE_THAT(computed.imag(), WithinRel(fp_t(expected_voigt[ia][iv].imag()), FP(1e-3)));
        }
    }
}

TEST_CASE("Voigt Interp", "[voigt_interp]") {
    yakl::init();
    {
        VoigtProfile<fp_t> prof(
            VoigtProfile<fp_t>::Linspace{FP(0.0), FP(10.0), 11},
            VoigtProfile<fp_t>::Linspace{FP(-20.0), FP(20.0), 11}
        );

        Fp2d samples("samples", 10, 10);
        parallel_for(
            SimpleBounds<1>(1),
            YAKL_LAMBDA (int _) {
                constexpr f32 as[] = {FP(0.0), FP(1.0), FP(1.5), FP(2.0), FP(3.8)};
                constexpr f32 vs[] = {FP(0.0), FP(2.0), FP(3.0), FP(4.0), FP(-11.4)};

                for (int ia = 0; ia < sizeof(as) / sizeof(as[0]); ++ia) {
                    for (int iv = 0; iv < sizeof(vs) / sizeof(vs[0]);  ++iv) {
                        fp_t a = as[ia];
                        fp_t v = vs[iv];
                        samples(ia, iv) = prof(a, v);
                    }
                }
            }
        );
        Fp2dHost test = samples.createHostCopy();

        // NOTE(cmo): Test grid directly, with complex (Voigt-Faraday)
        REQUIRE_THAT(test(0, 0), WithinRel(humlicek_voigt(FP(0.0), FP(0.0)).real(), FP(1e-6)));
        REQUIRE_THAT(test(1, 3), WithinRel(humlicek_voigt(FP(1.0), FP(4.0)).real(), FP(1e-6)));
        fp_t sample_a = humlicek_voigt(FP(1.0), FP(4.0)).real();
        fp_t sample_b = humlicek_voigt(FP(2.0), FP(4.0)).real();
        REQUIRE_THAT(test(2, 3), WithinRel(FP(0.5) * (sample_a + sample_b), FP(1e-6)));
        // NOTE(cmo): The grid is a bit coarse for accuracy, but it gets better if the grid is ramped up.
        REQUIRE_THAT(test(4, 4), WithinRel(humlicek_voigt(FP(3.8), FP(-11.4)).real(), FP(1e-1)));

        VoigtProfile<f64, true> complex_prof(
            VoigtProfile<f64>::Linspace{0.0, 15.0, 11},
            VoigtProfile<f64>::Linspace{-600.0, 600.0, 11}
        );

        yakl::Array<decltype(complex_prof)::Voigt_t, 2, yakl::memDevice> complex_dev("complex samples", n_a, n_v);
        YAKL_SCOPE(as_local, a);
        YAKL_SCOPE(vs_local, v);
        parallel_for(
            SimpleBounds<2>(n_a, n_v),
            YAKL_LAMBDA (int ia, int iv) {
                complex_dev(ia, iv) = complex_prof(as_local[ia], vs_local[iv]);
            }
        );
        auto complex_host = complex_dev.createHostCopy();
        for (int ia = 0; ia < n_a; ++ia) {
            for (int iv = 0; iv < n_v; ++iv) {
                REQUIRE_THAT(complex_host(ia, iv).real(), WithinRel(expected_voigt[ia][iv].real(), 1e-3));
                REQUIRE_THAT(complex_host(ia, iv).imag(), WithinRel(expected_voigt[ia][iv].imag(), 1e-3));
            }
        }

        // Check the object throws if we try to access from the CPU (in a cuda-like world)
    #if defined(YAKL_ARCH_CUDA) || defined(YAKL_ARCH_HIP) || defined(YAKL_ARCH_SYCL)
        REQUIRE_THROWS(prof(FP(1.0), FP(2.0)));
    #endif
        // But works if constructed on the CPU
        VoigtProfile<fp_t, false, yakl::memHost> prof_cpu(
            VoigtProfile<fp_t>::Linspace{FP(0.0), FP(10.0), 101},
            VoigtProfile<fp_t>::Linspace{FP(-20.0), FP(20.0), 101}
        );
        REQUIRE_THAT(prof_cpu(FP(1.0), FP(2.0)), WithinRel(humlicek_voigt(FP(1.0), FP(2.0)).real(), FP(1e-2)));
    }
    yakl::finalize();
}