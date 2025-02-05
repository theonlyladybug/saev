-- https://css-tricks.com/almanac/properties/i/image-rendering/


module Semseg exposing (..)

import Array
import Browser
import Browser.Navigation
import Dict
import File
import File.Select
import Gradio
import Html exposing (div)
import Html.Attributes exposing (class)
import Html.Events
import Http
import Json.Decode as D
import Json.Encode as E
import Parser exposing ((|.), (|=))
import Random
import Requests
import Set
import Task
import Url
import Url.Builder
import Url.Parser exposing ((</>), (<?>))
import Url.Parser.Query


main =
    Browser.application
        { init = init
        , view = view
        , update = update
        , subscriptions = \model -> Sub.none
        , onUrlRequest = onUrlRequest
        , onUrlChange = onUrlChange
        }



-- MESSAGE


type Msg
    = NoOp
    | SetUrl Int
    | SetExample Int
    | GetRandomExample
    | HoverPatch Int
    | ResetHoveredPatch
    | ToggleSelectedPatch Int
    | ResetSelectedPatches
    | SetSlider Int String
      -- API responses
    | GotExample Requests.Id (Result Gradio.Error Example)
    | GotSaeLatents Requests.Id (Result Gradio.Error (List SaeLatent))
    | GotOrigPreds Requests.Id (Result Gradio.Error Example)
    | GotModPreds Requests.Id (Result Gradio.Error Example)
    | ImageUploader ImageUploaderMsg


type ImageUploaderMsg
    = Upload
    | DragEnter
    | DragLeave
    | GotFile File.File
    | GotPreview String



-- MODEL


type alias Model =
    { key : Browser.Navigation.Key
    , ade20kIndex : Maybe Int
    , example : Requests.Requested Example
    , imageUploaderHover : Bool
    , hoveredPatchIndex : Maybe Int
    , selectedPatchIndices : Set.Set Int
    , saeLatents : Requests.Requested (List SaeLatent)

    -- Semantic segmenations
    , origPreds : Requests.Requested Example
    , modPreds : Requests.Requested Example

    -- UI
    , sliders : Dict.Dict Int Float

    -- API
    , gradio : Gradio.Config
    , exampleReqId : Requests.Id
    , saeLatentsReqId : Requests.Id
    , origPredsReqId : Requests.Id
    , modPredsReqId : Requests.Id
    }


type alias VitKey =
    String


type alias SaeLatent =
    { latent : Int
    , examples : List HighlightedExample
    }


type alias Example =
    { image : Gradio.Base64Image
    , labels : Gradio.Base64Image
    , classes : Set.Set Int
    }


type alias HighlightedExample =
    { original : Gradio.Base64Image
    , highlighted : Gradio.Base64Image
    , labels : Gradio.Base64Image
    , classes : Set.Set Int
    }


init : () -> Url.Url -> Browser.Navigation.Key -> ( Model, Cmd Msg )
init _ url key =
    let
        gradio =
            { host = "https://samuelstevens-saev-semantic-segmentation.hf.space" }

        ( ade20kIndex, cmd ) =
            case Maybe.andThen .index (Url.Parser.parse urlParser url) of
                Just index ->
                    ( Just index
                    , getExample gradio
                        (Requests.next Requests.init)
                        index
                    )

                Nothing ->
                    ( Nothing, Random.generate SetUrl randomExample )

        model =
            { key = key
            , ade20kIndex = ade20kIndex

            -- Image to segment.
            , example = Requests.Initial
            , imageUploaderHover = False

            -- Patches
            , hoveredPatchIndex = Nothing
            , selectedPatchIndices = Set.empty

            --
            , saeLatents = Requests.Initial

            --
            , origPreds = Requests.Initial
            , modPreds = Requests.Initial

            -- UI
            , sliders = Dict.empty

            -- API
            , gradio = gradio
            , exampleReqId = Requests.init
            , saeLatentsReqId = Requests.init
            , origPredsReqId = Requests.init
            , modPredsReqId = Requests.init
            }
    in
    ( model, cmd )



-- UPDATE


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        NoOp ->
            ( model, Cmd.none )

        SetUrl i ->
            let
                url =
                    Url.Builder.relative [] [ Url.Builder.int "example" i ]
            in
            ( model, Browser.Navigation.pushUrl model.key url )

        SetExample i ->
            let
                index =
                    modBy nImages i

                exampleReqId =
                    Requests.next model.exampleReqId
            in
            ( { model
                | ade20kIndex = Just index
                , example = Requests.Loading
                , selectedPatchIndices = Set.empty
                , saeLatents = Requests.Initial
                , origPreds = Requests.Loading
                , modPreds = Requests.Initial
                , exampleReqId = exampleReqId
              }
            , getExample model.gradio exampleReqId index
            )

        GetRandomExample ->
            ( { model
                | example = Requests.Loading
                , selectedPatchIndices = Set.empty
                , saeLatents = Requests.Initial
                , origPreds = Requests.Loading
                , modPreds = Requests.Initial
              }
            , Random.generate SetUrl randomExample
            )

        HoverPatch i ->
            ( { model | hoveredPatchIndex = Just i }, Cmd.none )

        ResetHoveredPatch ->
            ( { model | hoveredPatchIndex = Nothing }, Cmd.none )

        ToggleSelectedPatch i ->
            let
                patchIndices =
                    if Set.member i model.selectedPatchIndices then
                        Set.remove i model.selectedPatchIndices

                    else
                        Set.insert i model.selectedPatchIndices

                saeLatentsReqId =
                    Requests.next model.saeLatentsReqId

                cmd =
                    case model.example of
                        Requests.Loaded { image } ->
                            getSaeLatents model.gradio
                                saeLatentsReqId
                                image
                                patchIndices

                        _ ->
                            Cmd.none

                saeLatents =
                    if Set.isEmpty patchIndices then
                        Requests.Initial

                    else
                        Requests.Loading
            in
            ( { model
                | selectedPatchIndices = patchIndices
                , saeLatents = saeLatents
                , modPreds = Requests.Initial
                , saeLatentsReqId = saeLatentsReqId
              }
            , cmd
            )

        ResetSelectedPatches ->
            ( { model
                | selectedPatchIndices = Set.empty
                , saeLatents = Requests.Initial
                , modPreds = Requests.Initial
              }
            , Cmd.none
            )

        SetSlider i str ->
            case String.toFloat str of
                Just f ->
                    let
                        sliders =
                            Dict.insert i f model.sliders

                        modPredsReqId =
                            Requests.next model.modPredsReqId

                        cmd =
                            case model.example of
                                Requests.Loaded { image } ->
                                    getModPreds model.gradio
                                        modPredsReqId
                                        image
                                        sliders

                                _ ->
                                    Cmd.none
                    in
                    ( { model | sliders = sliders }
                    , cmd
                    )

                Nothing ->
                    ( model, Cmd.none )

        GotExample id result ->
            if Requests.isStale id model.exampleReqId then
                ( model, Cmd.none )

            else
                case result of
                    Ok example ->
                        let
                            origPredsReqId =
                                Requests.next model.origPredsReqId
                        in
                        ( { model
                            | example = Requests.Loaded example
                            , origPredsReqId = origPredsReqId
                          }
                        , getOrigPreds model.gradio origPredsReqId example.image
                        )

                    Err err ->
                        ( { model | example = Requests.Failed (explainGradioError err) }
                        , Cmd.none
                        )

        GotSaeLatents id result ->
            if Requests.isStale id model.saeLatentsReqId then
                ( model, Cmd.none )

            else
                case result of
                    Ok latents ->
                        let
                            sliders =
                                latents
                                    |> List.map .latent
                                    |> List.map (\latent -> ( latent, 0.0 ))
                                    |> Dict.fromList
                        in
                        ( { model | saeLatents = Requests.Loaded latents, sliders = sliders }, Cmd.none )

                    Err err ->
                        ( { model | saeLatents = Requests.Failed (explainGradioError err) }, Cmd.none )

        GotOrigPreds id result ->
            if Requests.isStale id model.origPredsReqId then
                ( model, Cmd.none )

            else
                case result of
                    Ok preds ->
                        ( { model | origPreds = Requests.Loaded preds }, Cmd.none )

                    Err err ->
                        ( { model
                            | origPreds = Requests.Failed (explainGradioError err)
                          }
                        , Cmd.none
                        )

        GotModPreds id result ->
            if Requests.isStale id model.modPredsReqId then
                ( model, Cmd.none )

            else
                case result of
                    Ok preds ->
                        ( { model | modPreds = Requests.Loaded preds }, Cmd.none )

                    Err err ->
                        ( { model
                            | modPreds = Requests.Failed (explainGradioError err)
                          }
                        , Cmd.none
                        )

        ImageUploader imageMsg ->
            imageUploaderUpdate model imageMsg


imageUploaderUpdate : Model -> ImageUploaderMsg -> ( Model, Cmd Msg )
imageUploaderUpdate model msg =
    case msg of
        Upload ->
            ( model, File.Select.file [ "image/*" ] (GotFile >> ImageUploader) )

        DragEnter ->
            ( { model | imageUploaderHover = True }, Cmd.none )

        DragLeave ->
            ( { model | imageUploaderHover = False }, Cmd.none )

        GotFile file ->
            ( { model | imageUploaderHover = False }
            , Task.perform (GotPreview >> ImageUploader) <| File.toUrl file
            )

        GotPreview preview ->
            case Gradio.base64Image preview of
                Just image ->
                    let
                        origPredsReqId =
                            Requests.next model.origPredsReqId
                    in
                    ( { model
                        | example =
                            Requests.Loaded
                                { image = image
                                , labels = Gradio.base64ImageEmpty
                                , classes = Set.empty
                                }
                        , selectedPatchIndices = Set.empty
                        , saeLatents = Requests.Initial
                        , origPreds = Requests.Loading
                        , modPreds = Requests.Initial
                        , origPredsReqId = origPredsReqId
                      }
                    , getOrigPreds model.gradio origPredsReqId image
                    )

                Nothing ->
                    ( { model | example = Requests.Failed "Uploaded image was not base64." }
                    , Cmd.none
                    )


onUrlRequest : Browser.UrlRequest -> Msg
onUrlRequest request =
    NoOp


onUrlChange : Url.Url -> Msg
onUrlChange url =
    Url.Parser.parse urlParser url
        |> Maybe.andThen .index
        |> Maybe.withDefault 0
        |> SetExample


type alias QueryParams =
    { index : Maybe Int
    }


urlParser : Url.Parser.Parser (QueryParams -> a) a
urlParser =
    -- Need to change this when I deploy it.
    Url.Parser.s "SAE-V"
        </> Url.Parser.s "demos"
        </> Url.Parser.s "semseg"
        <?> Url.Parser.Query.int "example"
        |> Url.Parser.map QueryParams



-- API


explainGradioError : Gradio.Error -> String
explainGradioError err =
    case err of
        Gradio.NetworkError msg ->
            "Network error: " ++ msg

        Gradio.JsonError msg ->
            "Error decoding JSON: " ++ msg

        Gradio.ParsingError msg ->
            "Error parsing API response: " ++ msg

        Gradio.ApiError msg ->
            "Error in the API: " ++ msg


encodeArgs : List E.Value -> E.Value
encodeArgs args =
    E.object [ ( "data", E.list identity args ) ]


getExample : Gradio.Config -> Requests.Id -> Int -> Cmd Msg
getExample cfg id img =
    Gradio.get cfg
        "get-img"
        [ E.int img ]
        (Gradio.decodeOne exampleDecoder)
        (GotExample id)


getOrigPreds : Gradio.Config -> Requests.Id -> Gradio.Base64Image -> Cmd Msg
getOrigPreds cfg id img =
    Gradio.get cfg
        "get-orig-preds"
        [ Gradio.encodeImg img ]
        (Gradio.decodeOne exampleDecoder)
        (GotOrigPreds id)


getModPreds : Gradio.Config -> Requests.Id -> Gradio.Base64Image -> Dict.Dict Int Float -> Cmd Msg
getModPreds cfg id img sliders =
    Gradio.get cfg
        "get-mod-preds"
        [ Gradio.encodeImg img
        , E.dict String.fromInt E.float sliders
        ]
        (Gradio.decodeOne exampleDecoder)
        (GotModPreds id)


getSaeLatents : Gradio.Config -> Requests.Id -> Gradio.Base64Image -> Set.Set Int -> Cmd Msg
getSaeLatents cfg id img patches =
    Gradio.get cfg
        "get-sae-latents"
        [ Gradio.encodeImg img
        , Set.toList patches |> E.list E.int
        ]
        (Gradio.decodeOne
            (D.list
                (D.map2 SaeLatent
                    (D.field "latent" D.int)
                    (D.field "examples" (D.list highlightedExampleDecoder))
                )
            )
        )
        (GotSaeLatents id)


exampleDecoder : D.Decoder Example
exampleDecoder =
    D.map3 Example
        (D.field "orig_url" Gradio.base64ImageDecoder)
        (D.field "seg_url" Gradio.base64ImageDecoder)
        (D.field "classes" (D.list D.int |> D.map Set.fromList))


highlightedExampleDecoder : D.Decoder HighlightedExample
highlightedExampleDecoder =
    D.map4 HighlightedExample
        (D.field "orig_url" Gradio.base64ImageDecoder)
        (D.field "highlighted_url" Gradio.base64ImageDecoder)
        (D.field "seg_url" Gradio.base64ImageDecoder)
        (D.field "classes" (D.list D.int |> D.map Set.fromList))


imgUrlDecoder : D.Decoder String
imgUrlDecoder =
    D.field "url" D.string
        |> D.map (String.replace "gradio_api/gradio_api" "gradio_api")
        |> D.map (String.replace "gra/gradio_api" "gradio_api")
        |> D.map (String.replace "gradio_ap/gradio_api" "gradio_api")
        |> D.map (String.replace "gradio_api/ca/gradio_api" "gradio_api")
        |> D.map (String.replace "gradio_api/call/gradio_api" "gradio_api")



-- VIEW


view : Model -> Browser.Document Msg
view model =
    { title = "Semantic Segmentation"
    , body =
        [ Html.header [] []
        , Html.main_
            [ class "w-full min-h-screen p-1 md:p-2 lg:p-4 bg-gray-50 space-y-4" ]
            [ Html.h2
                []
                [ Html.text "SAEs for Scientifically Rigorous Interpretation of Semantic Segmentation Models" ]
            , viewControls model.ade20kIndex
            , div
                [ class "flex flex-row" ]
                [ div
                    [ class "grid lg:grid-cols-[336px_336px] gap-1" ]
                    [ viewGriddedImage model
                        (Requests.map .image model.example)
                        "Input Image"
                        "Wait just a second..."
                    , viewGriddedImage model
                        (Requests.map .labels model.example)
                        "True Labels"
                        "Wait just a second..."
                    , viewGriddedImage model
                        (Requests.map .labels model.origPreds)
                        "Predicted Segmentation"
                        "Wait just a second..."
                    , viewGriddedImage model
                        (Requests.map .labels model.modPreds)
                        "Modified Segmentation"
                        (if Set.isEmpty model.selectedPatchIndices then
                            "Click on the image to explain model predictions."

                         else
                            "Modify the ViT's representations using the sliders."
                        )
                    ]
                , div
                    [ class "" ]
                    [ viewSaeLatents
                        model.selectedPatchIndices
                        model.saeLatents
                        model.sliders
                    , Set.empty
                        |> Set.union (getClasses model.example)
                        |> Set.union (getClasses model.origPreds)
                        |> Set.union (getClasses model.modPreds)
                        |> viewLegend
                    ]
                ]
            ]
        ]
    }


viewErr : String -> Html.Html Msg
viewErr err =
    div
        [ class "relative rounded-lg border border-red-200 bg-red-50 p-4 m-4" ]
        [ Html.button
            []
            []
        , Html.h3
            [ class "font-bold text-red-800" ]
            [ Html.text "Error" ]
        , Html.p
            [ class "text-red-700" ]
            [ Html.text err ]
        ]


viewControls : Maybe Int -> Html.Html Msg
viewControls ade20kIndex =
    let
        ( prevButton, nextButton ) =
            case ade20kIndex of
                Just i ->
                    ( viewButton (SetUrl (i - 1)) "Previous" True
                    , viewButton (SetUrl (i + 1)) "Next" True
                    )

                Nothing ->
                    ( viewButton NoOp "Previous" False
                    , viewButton NoOp "Next" False
                    )
    in
    div
        [ class "flex flex-row gap-2" ]
        [ prevButton
        , viewButton GetRandomExample "Random" True
        , nextButton
        , viewButton (ImageUploader Upload) "Upload Image" True

        -- , viewButton (ResetPatches) "Reset Patches" True
        ]


viewButton : Msg -> String -> Bool -> Html.Html Msg
viewButton onClick title enabled =
    Html.button
        [ Html.Events.onClick onClick
        , Html.Attributes.disabled (not enabled)
        , class "flex-1 rounded-lg px-2 py-1 transition-colors"
        , class "border border-sky-300 hover:border-sky-400"
        , class "bg-sky-100 hover:bg-sky-200"
        , class "text-gray-700 hover:text-gray-900"
        , class "focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2"
        , class "active:bg-gray-300"
        , class "disabled:bg-gray-100 disabled:border-gray-200 disabled:text-gray-400"
        , class "disabled:cursor-not-allowed"
        , class "disabled:hover:bg-gray-100 disabled:hover:border-gray-200 disabled:hover:text-gray-400"
        ]
        [ Html.text title ]


viewGriddedImage : Model -> Requests.Requested Gradio.Base64Image -> String -> String -> Html.Html Msg
viewGriddedImage model reqImage title callToAction =
    case reqImage of
        Requests.Initial ->
            div
                []
                [ Html.p
                    [ class "text-center" ]
                    [ Html.text title ]
                , Html.p
                    [ class "italic" ]
                    [ Html.text callToAction ]
                ]

        Requests.Loading ->
            div
                []
                [ Html.p
                    [ class "text-center" ]
                    [ Html.text title ]
                , Html.p
                    [ class "italic" ]
                    [ Html.text "Loading..." ]
                ]

        Requests.Failed err ->
            viewErr err

        Requests.Loaded image ->
            div []
                [ Html.p
                    [ class "text-center" ]
                    [ Html.text title ]
                , div
                    [ class "relative inline-block" ]
                    [ div
                        [ class "absolute grid"
                        , class "grid-rows-[repeat(16,_14px)] grid-cols-[repeat(16,_14px)]"
                        , class "lg:grid-rows-[repeat(16,_21px)] lg:grid-cols-[repeat(16,_21px)]"
                        ]
                        (List.map
                            (viewGridCell model.hoveredPatchIndex model.selectedPatchIndices)
                            (List.range 0 255)
                        )
                    , Html.img
                        [ class "block"
                        , class "w-[224px] h-[224px]"
                        , class "lg:w-[336px] lg:h-[336px]"
                        , Html.Attributes.src (Gradio.base64ImageToString image)
                        , Html.Attributes.style "image-rendering" "pixelated"
                        ]
                        []
                    ]
                ]


viewGridCell : Maybe Int -> Set.Set Int -> Int -> Html.Html Msg
viewGridCell hovered selected self =
    let
        classes =
            (case hovered of
                Just h ->
                    if h == self then
                        [ "border-2 border-rose-600 border-dashed" ]

                    else
                        []

                Nothing ->
                    []
            )
                ++ (if Set.member self selected then
                        [ "bg-rose-600/50" ]

                    else
                        []
                   )
    in
    div
        ([ class "w-[14px] h-[14px] lg:w-[21px] lg:h-[21px]"
         , Html.Events.onMouseEnter (HoverPatch self)
         , Html.Events.onMouseLeave ResetHoveredPatch
         , Html.Events.onClick (ToggleSelectedPatch self)
         ]
            ++ List.map class classes
        )
        []


viewLegend : Set.Set Int -> Html.Html Msg
viewLegend classes =
    div
        [ Html.Attributes.id "legend" ]
        [ Html.p [] [ Html.text "Legend" ]
        , div
            []
            (Set.toList classes
                |> List.sort
                |> List.filter (\x -> x > 0)
                |> List.map viewClassIcon
            )
        ]


viewClassIcon : Int -> Html.Html Msg
viewClassIcon cls =
    let
        color =
            case Array.get (cls - 1) colors of
                Just ( r, g, b ) ->
                    "rgb(" ++ String.fromInt r ++ " " ++ String.fromInt g ++ " " ++ String.fromInt b ++ ")"

                Nothing ->
                    "red"

        classname =
            case Array.get (cls - 1) classnames of
                Just names ->
                    names |> List.take 2 |> String.join ", "

                Nothing ->
                    "no classname found"
    in
    div
        [ class "flex flex-row gap-1 items-center" ]
        [ Html.span
            [ class "w-4 h-4"
            , Html.Attributes.style "background-color" color
            ]
            []
        , Html.span
            []
            [ Html.text (classname ++ " (class " ++ String.fromInt cls ++ ")") ]
        ]


viewSaeLatents : Set.Set Int -> Requests.Requested (List SaeLatent) -> Dict.Dict Int Float -> Html.Html Msg
viewSaeLatents selected requestedLatents values =
    case requestedLatents of
        Requests.Initial ->
            Html.p []
                [ Html.text "Click on the image above to explain model predictions." ]

        Requests.Loading ->
            Html.p []
                [ Html.text "Loading similar patches..." ]

        Requests.Failed err ->
            viewErr err

        Requests.Loaded latents ->
            div []
                (List.filterMap
                    (\latent -> Maybe.map (\f -> viewSaeLatent latent f) (Dict.get latent.latent values))
                    latents
                )


viewSaeLatent : SaeLatent -> Float -> Html.Html Msg
viewSaeLatent latent value =
    div
        []
        [ div
            [ class "grid grid-cols-4" ]
            (List.map viewHighlightedExample latent.examples)
        , div
            [ class "" ]
            [ Html.input
                [ Html.Attributes.type_ "range"
                , Html.Attributes.min "-10"
                , Html.Attributes.max "10"
                , Html.Attributes.step "0.2"
                , Html.Attributes.value (String.fromFloat value)
                , Html.Events.onInput (SetSlider latent.latent)
                ]
                []
            , Html.p
                [ class "gap-1" ]
                [ Html.span [] [ Html.text ("Latent 24K/" ++ String.fromInt latent.latent) ]
                , Html.span [] [ Html.text ("Value:" ++ String.fromFloat value) ]
                ]
            ]
        ]


viewHighlightedExample : HighlightedExample -> Html.Html Msg
viewHighlightedExample { original, highlighted } =
    Html.img
        [ Html.Attributes.src (Gradio.base64ImageToString highlighted)
        , class "max-w-36 h-auto"
        ]
        []



-- HELPERS


getClasses : Requests.Requested Example -> Set.Set Int
getClasses example =
    case example of
        Requests.Loading ->
            Set.empty

        Requests.Initial ->
            Set.empty

        Requests.Failed _ ->
            Set.empty

        Requests.Loaded { classes } ->
            classes



-- CONSTANTS


nImages =
    2000


nSaeLatents =
    3


nSaeExamplesPerLatent =
    4


randomExample : Random.Generator Int
randomExample =
    Random.int 0 (nImages - 1)


colors : Array.Array ( Int, Int, Int )
colors =
    [ ( 51, 0, 0 ), ( 204, 0, 102 ), ( 0, 255, 0 ), ( 102, 51, 51 ), ( 153, 204, 51 ), ( 51, 51, 153 ), ( 102, 0, 51 ), ( 153, 153, 0 ), ( 51, 102, 204 ), ( 204, 255, 0 ), ( 204, 102, 0 ), ( 204, 255, 153 ), ( 102, 102, 255 ), ( 255, 204, 255 ), ( 51, 255, 0 ), ( 0, 102, 51 ), ( 102, 102, 0 ), ( 0, 0, 255 ), ( 255, 153, 204 ), ( 204, 204, 0 ), ( 0, 153, 153 ), ( 153, 102, 204 ), ( 255, 204, 0 ), ( 204, 204, 153 ), ( 255, 51, 0 ), ( 51, 51, 0 ), ( 153, 51, 51 ), ( 0, 0, 102 ), ( 102, 255, 204 ), ( 204, 51, 255 ), ( 255, 204, 204 ), ( 0, 0, 153 ), ( 0, 102, 153 ), ( 153, 0, 51 ), ( 51, 51, 102 ), ( 255, 153, 0 ), ( 204, 153, 0 ), ( 153, 102, 153 ), ( 51, 204, 204 ), ( 51, 51, 255 ), ( 153, 204, 102 ), ( 102, 204, 153 ), ( 153, 153, 204 ), ( 0, 51, 204 ), ( 204, 204, 102 ), ( 0, 51, 153 ), ( 0, 102, 0 ), ( 51, 0, 102 ), ( 153, 255, 0 ), ( 153, 255, 102 ), ( 102, 102, 51 ), ( 153, 0, 255 ), ( 204, 255, 102 ), ( 102, 0, 255 ), ( 255, 204, 153 ), ( 102, 51, 0 ), ( 102, 204, 102 ), ( 0, 102, 204 ), ( 51, 204, 0 ), ( 255, 102, 102 ), ( 153, 255, 204 ), ( 51, 204, 51 ), ( 0, 0, 0 ), ( 255, 0, 255 ), ( 153, 0, 153 ), ( 255, 204, 51 ), ( 51, 0, 51 ), ( 102, 204, 255 ), ( 153, 204, 153 ), ( 153, 102, 0 ), ( 102, 204, 204 ), ( 204, 204, 204 ), ( 255, 0, 0 ), ( 255, 255, 51 ), ( 0, 255, 102 ), ( 204, 153, 102 ), ( 204, 153, 153 ), ( 102, 51, 153 ), ( 51, 102, 0 ), ( 204, 51, 153 ), ( 153, 51, 255 ), ( 102, 0, 204 ), ( 204, 102, 153 ), ( 204, 0, 204 ), ( 102, 51, 102 ), ( 0, 153, 51 ), ( 153, 153, 51 ), ( 255, 102, 0 ), ( 255, 153, 153 ), ( 153, 0, 102 ), ( 51, 204, 255 ), ( 102, 255, 102 ), ( 255, 255, 204 ), ( 51, 51, 204 ), ( 153, 102, 51 ), ( 153, 153, 255 ), ( 51, 153, 0 ), ( 204, 0, 255 ), ( 102, 255, 0 ), ( 153, 102, 255 ), ( 204, 102, 255 ), ( 204, 0, 0 ), ( 102, 153, 255 ), ( 204, 102, 204 ), ( 204, 51, 102 ), ( 0, 255, 153 ), ( 153, 204, 204 ), ( 255, 0, 102 ), ( 102, 51, 204 ), ( 255, 51, 204 ), ( 51, 204, 153 ), ( 153, 153, 102 ), ( 153, 204, 0 ), ( 153, 102, 102 ), ( 204, 153, 255 ), ( 153, 0, 204 ), ( 102, 0, 0 ), ( 255, 51, 255 ), ( 0, 204, 153 ), ( 255, 153, 51 ), ( 0, 255, 204 ), ( 51, 102, 153 ), ( 255, 51, 51 ), ( 102, 255, 51 ), ( 0, 0, 204 ), ( 102, 255, 153 ), ( 0, 204, 255 ), ( 0, 102, 102 ), ( 102, 51, 255 ), ( 255, 0, 204 ), ( 51, 255, 153 ), ( 204, 0, 51 ), ( 153, 51, 204 ), ( 204, 102, 51 ), ( 255, 255, 0 ), ( 51, 51, 51 ), ( 0, 153, 0 ), ( 51, 255, 102 ), ( 51, 102, 255 ), ( 102, 153, 0 ), ( 102, 153, 204 ), ( 51, 0, 255 ), ( 102, 153, 153 ), ( 153, 51, 102 ), ( 204, 255, 51 ), ( 204, 204, 51 ), ( 0, 204, 51 ), ( 255, 102, 153 ), ( 204, 102, 102 ), ( 102, 0, 102 ), ( 51, 153, 204 ), ( 255, 255, 255 ), ( 0, 102, 255 ), ( 51, 102, 51 ), ( 204, 0, 153 ), ( 102, 153, 102 ), ( 102, 0, 153 ), ( 153, 255, 153 ), ( 0, 153, 102 ), ( 102, 204, 0 ), ( 0, 255, 51 ), ( 153, 204, 255 ), ( 153, 51, 153 ), ( 0, 51, 255 ), ( 51, 255, 51 ), ( 255, 102, 51 ), ( 102, 102, 204 ), ( 102, 153, 51 ), ( 0, 204, 0 ), ( 102, 204, 51 ), ( 255, 102, 255 ), ( 255, 204, 102 ), ( 102, 102, 102 ), ( 255, 102, 204 ), ( 51, 0, 153 ), ( 255, 0, 51 ), ( 102, 102, 153 ), ( 255, 153, 102 ), ( 204, 255, 204 ), ( 51, 0, 204 ), ( 0, 0, 51 ), ( 51, 255, 255 ), ( 204, 51, 0 ), ( 153, 51, 0 ), ( 51, 153, 102 ), ( 102, 255, 255 ), ( 255, 153, 255 ), ( 204, 255, 255 ), ( 204, 153, 204 ), ( 255, 0, 153 ), ( 51, 102, 102 ), ( 153, 255, 255 ), ( 255, 255, 153 ), ( 204, 51, 204 ), ( 153, 153, 153 ), ( 51, 153, 255 ), ( 51, 153, 51 ), ( 0, 153, 255 ), ( 0, 51, 51 ), ( 0, 51, 102 ), ( 153, 0, 0 ), ( 204, 51, 51 ), ( 0, 153, 204 ), ( 153, 255, 51 ), ( 255, 255, 102 ), ( 204, 204, 255 ), ( 0, 204, 102 ), ( 255, 51, 102 ), ( 0, 255, 255 ), ( 51, 153, 153 ), ( 51, 204, 102 ), ( 51, 255, 204 ), ( 255, 51, 153 ), ( 0, 51, 0 ), ( 0, 204, 204 ), ( 204, 153, 51 ) ]
        |> Array.fromList
        -- Sky
        |> Array.set 2 ( 201, 249, 255 )
        -- Tree
        |> Array.set 4 ( 151, 204, 4 )
        -- Mountain
        |> Array.set 16 ( 54, 48, 32 )
        -- Water
        |> Array.set 21 ( 120, 202, 210 )
        -- Sea
        |> Array.set 26 ( 45, 125, 210 )
        -- Field
        |> Array.set 29 ( 116, 142, 84 )
        -- Sand
        |> Array.set 46 ( 238, 185, 2 )
        -- River
        |> Array.set 60 ( 72, 99, 156 )
        -- Palm, Palm Tree
        |> Array.set 72 ( 76, 46, 5 )
        -- Land, Ground
        |> Array.set 94 ( 12, 15, 10 )


classnames : Array.Array (List String)
classnames =
    Array.fromList
        [ [ "wall" ]
        , [ "building", "edifice" ]
        , [ "sky" ]
        , [ "floor", "flooring" ]
        , [ "tree" ]
        , [ "ceiling" ]
        , [ "road", "route" ]
        , [ "bed" ]
        , [ "windowpane", "window" ]
        , [ "grass" ]
        , [ "cabinet" ]
        , [ "sidewalk", "pavement" ]
        , [ "person", "individual" ]
        , [ "earth", "ground" ]
        , [ "door", "double door" ]
        , [ "table" ]
        , [ "mountain" ]
        , [ "plant", "flora" ]
        , [ "curtain", "drape" ]
        , [ "chair" ]
        , [ "car", "auto" ]
        , [ "water" ]
        , [ "painting", "picture" ]
        , [ "sofa", "couch" ]
        , [ "shelf" ]
        , [ "house" ]
        , [ "sea" ]
        , [ "mirror" ]
        , [ "rug", "carpet" ]
        , [ "field" ]
        , [ "armchair" ]
        , [ "seat" ]
        , [ "fence", "fencing" ]
        , [ "desk" ]
        , [ "rock", "stone" ]
        , [ "wardrobe", "closet" ]
        , [ "lamp" ]
        , [ "bathtub", "bathing tub" ]
        , [ "railing", "rail" ]
        , [ "cushion" ]
        , [ "base", "pedestal" ]
        , [ "box" ]
        , [ "column", "pillar" ]
        , [ "signboard", "sign" ]
        , [ "chest of drawers", "chest" ]
        , [ "counter" ]
        , [ "sand" ]
        , [ "sink" ]
        , [ "skyscraper" ]
        , [ "fireplace", "hearth" ]
        , [ "refrigerator", "icebox" ]
        , [ "grandstand", "covered stand" ]
        , [ "path" ]
        , [ "stairs", "steps" ]
        , [ "runway" ]
        , [ "case", "display case" ]
        , [ "pool table", "billiard table" ]
        , [ "pillow" ]
        , [ "screen door", "screen" ]
        , [ "stairway", "staircase" ]
        , [ "river" ]
        , [ "bridge", "span" ]
        , [ "bookcase" ]
        , [ "blind", "screen" ]
        , [ "coffee table", "cocktail table" ]
        , [ "toilet", "can" ]
        , [ "flower" ]
        , [ "book" ]
        , [ "hill" ]
        , [ "bench" ]
        , [ "countertop" ]
        , [ "stove", "kitchen stove" ]
        , [ "palm", "palm tree" ]
        , [ "kitchen island" ]
        , [ "computer", "computing machine" ]
        , [ "swivel chair" ]
        , [ "boat" ]
        , [ "bar" ]
        , [ "arcade machine" ]
        , [ "hovel", "hut" ]
        , [ "bus", "autobus" ]
        , [ "towel" ]
        , [ "light", "light source" ]
        , [ "truck", "motortruck" ]
        , [ "tower" ]
        , [ "chandelier", "pendant" ]
        , [ "awning", "sunshade" ]
        , [ "streetlight", "street lamp" ]
        , [ "booth", "cubicle" ]
        , [ "television receiver", "television" ]
        , [ "airplane", "aeroplane" ]
        , [ "dirt track" ]
        , [ "apparel", "wearing apparel" ]
        , [ "pole" ]
        , [ "land", "ground" ]
        , [ "bannister", "banister" ]
        , [ "escalator", "moving staircase" ]
        , [ "ottoman", "pouf" ]
        , [ "bottle" ]
        , [ "buffet", "counter" ]
        , [ "poster", "posting" ]
        , [ "stage" ]
        , [ "van" ]
        , [ "ship" ]
        , [ "fountain" ]
        , [ "conveyer belt", "conveyor belt" ]
        , [ "canopy" ]
        , [ "washer", "automatic washer" ]
        , [ "plaything", "toy" ]
        , [ "swimming pool", "swimming bath" ]
        , [ "stool" ]
        , [ "barrel", "cask" ]
        , [ "basket", "handbasket" ]
        , [ "waterfall", "falls" ]
        , [ "tent", "collapsible shelter" ]
        , [ "bag" ]
        , [ "minibike", "motorbike" ]
        , [ "cradle" ]
        , [ "oven" ]
        , [ "ball" ]
        , [ "food", "solid food" ]
        , [ "step", "stair" ]
        , [ "tank", "storage tank" ]
        , [ "trade name", "brand name" ]
        , [ "microwave", "microwave oven" ]
        , [ "pot", "flowerpot" ]
        , [ "animal", "animate being" ]
        , [ "bicycle", "bike" ]
        , [ "lake" ]
        , [ "dishwasher", "dish washer" ]
        , [ "screen", "silver screen" ]
        , [ "blanket", "cover" ]
        , [ "sculpture" ]
        , [ "hood", "exhaust hood" ]
        , [ "sconce" ]
        , [ "vase" ]
        , [ "traffic light", "traffic signal" ]
        , [ "tray" ]
        , [ "ashcan", "trash can" ]
        , [ "fan" ]
        , [ "pier", "wharf" ]
        , [ "crt screen" ]
        , [ "plate" ]
        , [ "monitor", "monitoring device" ]
        , [ "bulletin board", "notice board" ]
        , [ "shower" ]
        , [ "radiator" ]
        , [ "glass", "drinking glass" ]
        , [ "clock" ]
        , [ "flag" ]
        ]
